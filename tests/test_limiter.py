import asyncio
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from rate_limiter import RateLimit
from rate_limiter.exceptions import RetryLimitReachedError

log: logging.Logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_successful_execution():
    """Test that the function executes successfully when limit is not reached."""
    redis_mock = Mock()
    lua_mock = AsyncMock()
    lua_mock.return_value = [0, 1, 0]
    redis_mock.register_script.return_value = lua_mock

    rate_limit = RateLimit(
        redis=redis_mock,
        limit=5,
        window=10,
        retries=3,
        backoff_ms=100,
        backoff_factor=1.0,
    )

    executed = False

    async def my_fn():
        nonlocal executed
        executed = True
        return 42

    wrapped = rate_limit(fn=my_fn, key='test')
    result = await wrapped()
    assert executed
    assert result == 42


@pytest.mark.asyncio
async def test_rate_limit_hit_with_retry():
    """Test that retries are triggered when the rate limit is hit."""
    redis_mock = Mock()
    lua_mock = AsyncMock()
    # first two blocked (allowed=0), third allowed
    lua_mock.side_effect = [
        [1, 0, 100],
        [2, 0, 100],
        [3, 1, 0],
    ]
    redis_mock.register_script.return_value = lua_mock

    rate_limit = RateLimit(
        redis=redis_mock,
        limit=1,
        window=10,
        retries=3,
        backoff_ms=100,
        backoff_factor=1.0,
    )

    executed = False

    async def my_fn():
        nonlocal executed
        executed = True
        return 'done'

    sleep_mock = AsyncMock()
    bound = asyncio.sleep
    asyncio.sleep = sleep_mock  # temporarily override

    try:
        wrapped = rate_limit(fn=my_fn, key='test')
        result = await wrapped()
    finally:
        asyncio.sleep = bound

    assert executed
    assert result == 'done'
    assert sleep_mock.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_exceptions_logged(caplog):
    """Test that exceptions in retry_on_exceptions are logged and retried."""
    redis_mock = Mock()
    lua_mock = AsyncMock()
    lua_mock.return_value = [0, 1, 0]
    redis_mock.register_script.return_value = lua_mock

    rate_limit = RateLimit(
        redis=redis_mock,
        limit=1,
        window=10,
        retry_on_exceptions=(ValueError,),
        retries=2,
        backoff_ms=50,
        backoff_factor=1.0,
    )

    async def my_fn():
        raise ValueError('retry me')

    wrapped = rate_limit(fn=my_fn, key='test')
    caplog.set_level(logging.WARNING)
    with (
        pytest.raises(RetryLimitReachedError),
        patch('asyncio.sleep', new=AsyncMock()),
    ):
        result = await wrapped()

        assert result is None
        assert any('retrying' in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_unhandled_exception_stops():
    """Test that unhandled exceptions immediately stop the limiter."""
    redis_mock = Mock()
    lua_mock = AsyncMock()
    lua_mock.return_value = [0, 1, 0]
    redis_mock.register_script.return_value = lua_mock

    rate_limit = RateLimit(
        redis=redis_mock,
        limit=1,
        window=10,
        retries=2,
        backoff_ms=50,
        backoff_factor=1.0,
    )

    async def my_fn():
        raise RuntimeError('stop')

    wrapped = rate_limit(fn=my_fn, key='test')
    with (
        patch('asyncio.sleep', new=AsyncMock()),
        pytest.raises(RuntimeError),
    ):
        await wrapped()


@pytest.mark.asyncio
async def test_decorator_syntax_usage():
    """Test using the RateLimit object as a decorator with @ syntax."""
    redis_mock = Mock()
    lua_mock = AsyncMock()
    lua_mock.return_value = [0, 1, 0]
    redis_mock.register_script.return_value = lua_mock

    rate_limit = RateLimit(
        redis=redis_mock,
        limit=1,
        window=10,
        retries=2,
        backoff_ms=50,
        backoff_factor=1.0,
    )

    executed = False

    @rate_limit(key='test')
    async def my_fn():
        nonlocal executed
        executed = True
        return 'ok'

    sleep_mock = AsyncMock()
    bound = asyncio.sleep
    asyncio.sleep = sleep_mock

    try:
        result = await my_fn()
    finally:
        asyncio.sleep = bound

    assert executed
    assert result == 'ok'


@pytest.mark.asyncio
async def test_exponential_backoff_and_wait_ms():
    """Test that backoff uses wait_ms when provided by Lua script."""
    redis_mock = Mock()
    lua_mock = AsyncMock()

    # simulate first call blocked w/wait_ms, then allowed
    lua_mock.side_effect = [
        [1, 0, 500],
        [2, 1, 0],
    ]
    redis_mock.register_script.return_value = lua_mock

    rate_limit = RateLimit(
        redis=redis_mock,
        limit=1,
        window=2,
        retries=2,
        backoff_ms=100,
        backoff_factor=1.0,
    )

    call_order = []
    async def my_fn():
        call_order.append('executed')
        return 'done'

    wrapped = rate_limit(fn=my_fn, key='test')

    sleep_calls = []
    async def fake_sleep(duration: float):
        sleep_calls.append(duration)

    bound = asyncio.sleep
    asyncio.sleep = fake_sleep  # override

    try:
        result = await wrapped()
    finally:
        asyncio.sleep = bound

    assert result == 'done'
    # sleep should use wait_ms 500 (> backoff_ms 100)
    assert sleep_calls == [0.5]
    assert call_order == ['executed']


@pytest.mark.asyncio
async def test_high_rps_limit_concurrent(redis_mock):
    """Ensure that RateLimit correctly enforces 30 RPS under concurrent load."""
    limiter = RateLimit(redis=redis_mock, limit=30, window=1, retries=1)

    async def dummy_task(i: int):
        await asyncio.sleep(0)
        return f'ok-{i}'

    limited_task = limiter(dummy_task, key='high_rps:test')

    async def run_task(i: int):
        try:
            return await limited_task(i)
        except RetryLimitReachedError:
            return None

    # Launch 31 concurrent tasks
    tasks = [run_task(i) for i in range(31)]
    results = await asyncio.gather(*tasks)

    allowed = [r for r in results if r is not None]
    denied = [r for r in results if r is None]

    # Verify that at most 30 are allowed
    assert len(allowed) == 30
    assert len(denied) == 1

    # Ensure all allowed results are unique and sequential
    assert all(r.startswith('ok-') for r in allowed)
    assert len(set(allowed)) == len(allowed)
