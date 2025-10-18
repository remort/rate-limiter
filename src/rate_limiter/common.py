import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from redis.asyncio import Redis
from typing_extensions import ParamSpec

from rate_limiter.exceptions import RetryLimitReachedError

log: logging.Logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

type TargetFunction[T, **P] = Callable[P, Awaitable[T]]


SLIDING_WINDOW_LUA_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2]) * 1000
local limit = tonumber(ARGV[3])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
redis.call('ZADD', key, now, tostring(now))
local count = redis.call('ZCARD', key)
redis.call('EXPIRE', key, ARGV[2])

if count <= limit then
    return {count, 1, 0}
else
    local earliest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')[2]
    local wait = window - (now - earliest)
    return {count, 0, wait}
end
"""


@dataclass
class RateLimit:
    redis: Redis  # type: ignore
    limit: int
    window: int = 1
    retries: int = 3
    backoff_ms: int = 10
    backoff_factor: float = 1.0
    retry_on_exceptions: tuple[type[BaseException], ...] = ()
    logger: logging.Logger = log

    def __post_init__(self) -> None:
        self._lua_script = self.redis.register_script(SLIDING_WINDOW_LUA_SCRIPT)

    async def is_execution_allowed(self, key: str) -> tuple[bool, int]:
        now: int = int(time.time() * 1000)
        count_allowed = await self._lua_script(keys=[key], args=[now, self.window, self.limit])
        count, allowed, wait_ms = count_allowed
        self.logger.info(
            'Limiter stats. count: %s, allowed: %s, wait ms: %s',
            count,
            allowed,
            wait_ms,
        )
        return bool(allowed), int(wait_ms)

    def __call__[T, **P](
        self,
        fn: TargetFunction[T, P] | None = None,
        *,
        key: str,
    ) -> TargetFunction[T, P]:
        def decorator(inner_fn: TargetFunction[T, P]) -> TargetFunction[T, P]:
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                delay: float = self.backoff_ms
                for attempt in range(1, self.retries + 1):
                    try:
                        allowed, wait_ms = await self.is_execution_allowed(key)
                        if allowed:
                            return await inner_fn(*args, **kwargs)
                        else:
                            self.logger.info('Request is rate limited.')
                    except self.retry_on_exceptions as e:
                        self.logger.warning(
                            'Exception %s occurred during execution of %s, retrying. ' \
                            'Attempt %s/%s.',
                            e,
                            key,
                            attempt,
                            self.retries,
                        )
                    except Exception:
                        self.logger.exception(
                            'Unhandled exception in decorated function. Limiter stops.',
                        )
                        raise

                    sleep_time = max(delay, wait_ms)
                    self.logger.warning(
                        'Rate limit hit for %s. Attempt %s/%s. Retrying in %s ms.',
                        key,
                        attempt,
                        self.retries,
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time / 1000)
                    delay *= self.backoff_factor

                self.logger.error(
                    'All %s attempts exhausted for %s. Giving up.', self.retries, key,
                )
                raise RetryLimitReachedError('Attempts limit reached.')

            return wrapper

        return decorator(fn) if fn is not None else decorator  # type: ignore
