from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def redis_mock() -> Any:
    """
    Mocked Redis client for testing RateLimit behavior.

    The mocked register_script returns an async function that implements
    the same contract as the real Lua script:
      returns [count, allowed, wait_ms].
    """
    redis = MagicMock()
    # in-memory store: key -> list of timestamps (ms)
    redis._data = {}

    async def mock_lua_script(*, keys=None, args=None):
        """
        Emulate Lua sliding-window script.

        Args:
            keys: list with single redis key.
            args: [now_ms, window_seconds, limit]

        Returns:
            [count, allowed_flag (0|1), wait_ms]
        """
        key = keys[0]
        now = int(args[0])
        window_seconds = int(args[1])
        window_ms = window_seconds * 1000
        limit = int(args[2])

        # get or init timestamps list for key
        timestamps = redis._data.setdefault(key, [])

        # remove expired timestamps (<= now - window)
        cutoff = now - window_ms
        timestamps[:] = [t for t in timestamps if t > cutoff]

        # append current timestamp
        timestamps.append(now)

        # ensure timestamps sorted (they should be, but be defensive)
        timestamps.sort()

        count = len(timestamps)

        if count <= limit:
            allowed = 1
            wait_ms = 0
        else:
            allowed = 0
            # earliest timestamp in the window
            earliest = timestamps[0]
            wait_ms = window_ms - (now - earliest)
            wait_ms = max(wait_ms, 0)

        return [count, allowed, int(wait_ms)]

    # register_script should return an awaitable callable (the lua script)
    redis.register_script = MagicMock(return_value=mock_lua_script)

    return redis
