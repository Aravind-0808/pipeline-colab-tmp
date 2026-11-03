"""Counters that have to agree across processes: rate windows and slots.

Two shapes cover every limit the platform has:

- a **window**: at most N events per T seconds for a key — logins, uploads per
  minute;
- a **slot**: at most N holders at once for a key — uploads in flight, model
  calls in flight. A slot is leased, so a process that dies holding one gives it
  back when the lease runs out instead of shrinking the limit for ever.

Both live in Redis when it is configured, as small Lua scripts so each check is
one atomic step, and in this process's memory when it is not or when Redis does
not answer. The local versions are the same arithmetic over a dict.
"""

from __future__ import annotations

import asyncio
import math
import random
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager

from dfpipeline.redis import complain, get_redis

_PREFIX = "limits:"

# KEYS[1] window zset. ARGV: now, window_sec, limit, member.
# Returns -1 when the event is recorded, else seconds until there is room.
_WINDOW = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then
  local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
  return math.max(1, math.ceil(tonumber(oldest[2]) + window - now))
end
redis.call('ZADD', KEYS[1], now, ARGV[4])
redis.call('PEXPIRE', KEYS[1], math.ceil(window * 1000))
return -1
"""

# KEYS[1] window zset. ARGV: now, window_sec, limit. Like _WINDOW, but only
# looks: -1 when there is room, else seconds until there is.
_PEEK = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then
  local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
  return math.max(1, math.ceil(tonumber(oldest[2]) + window - now))
end
return -1
"""

# KEYS[1] slot zset. ARGV: now, lease_sec, max, token. Returns 1 when taken.
_TAKE = """
local now = tonumber(ARGV[1])
local lease = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - lease)
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[3]) then
  redis.call('ZADD', KEYS[1], now, ARGV[4])
  redis.call('PEXPIRE', KEYS[1], math.ceil(lease * 1000))
  return 1
end
return 0
"""

# KEYS[1] slot zset. ARGV: now, lease_sec. Live holders, stale ones dropped.
_HELD = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', tonumber(ARGV[1]) - tonumber(ARGV[2]))
return redis.call('ZCARD', KEYS[1])
"""


class SlotTimeout(Exception):
    """No slot came free within the wait."""


# ------------------------------------------------------------------ local

_lock = threading.Lock()
_windows: dict[str, list[float]] = defaultdict(list)
_slots: dict[str, dict[str, float]] = defaultdict(dict)


def _local_window(key: str, limit: int, window: float, at: float) -> int | None:
    with _lock:
        hits = _windows[key]
        hits[:] = [t for t in hits if at - t < window]
        if len(hits) >= limit:
            return max(1, math.ceil(hits[0] + window - at))
        hits.append(at)
        return None


def _local_take(key: str, limit: int, lease: float, token: str, at: float) -> bool:
    with _lock:
        held = _slots[key]
        for t, since in list(held.items()):
            if at - since >= lease:
                held.pop(t, None)
        if len(held) < limit:
            held[token] = at
            return True
        return False


def clear_local() -> None:
    """Forget every local counter. For tests."""
    with _lock:
        _windows.clear()
        _slots.clear()


# ----------------------------------------------------------------- windows


def window_hit(key: str, limit: int, window_sec: float) -> int | None:
    """Record one event against `key`, or say how long until there is room.

    Returns None when the event fits, else the seconds to wait (at least 1).
    """
    at = time.time()
    r = get_redis()
    if r is not None:
        try:
            wait = int(r.eval(_WINDOW, 1, _PREFIX + "w:" + key, at, window_sec, limit, uuid.uuid4().hex))
            return None if wait < 0 else wait
        except Exception as exc:
            complain("rate windows", exc)
    return _local_window(key, limit, window_sec, at)


def window_peek(key: str, limit: int, window_sec: float) -> int | None:
    """Whether `key` is over its limit, without recording anything.

    For limits that count only failures: look before the attempt, record
    after it fails.
    """
    at = time.time()
    r = get_redis()
    if r is not None:
        try:
            wait = int(r.eval(_PEEK, 1, _PREFIX + "w:" + key, at, window_sec, limit))
            return None if wait < 0 else wait
        except Exception as exc:
            complain("rate windows", exc)
    with _lock:
        hits = [t for t in _windows.get(key, []) if at - t < window_sec]
        if len(hits) >= limit:
            return max(1, math.ceil(hits[0] + window_sec - at))
        return None


def window_reset(key: str) -> None:
    """Forget a key's events — a successful login clears its failures."""
    with _lock:
        _windows.pop(key, None)
    r = get_redis()
    if r is not None:
        try:
            r.delete(_PREFIX + "w:" + key)
        except Exception as exc:
            complain("rate windows", exc)


# ------------------------------------------------------------------- slots


def try_take(key: str, limit: int, lease_sec: float) -> str | None:
    """Take a slot if one is free. Returns the token to give back, or None."""
    token = uuid.uuid4().hex
    at = time.time()
    r = get_redis()
    if r is not None:
        try:
            taken = int(r.eval(_TAKE, 1, _PREFIX + "s:" + key, at, lease_sec, limit, token))
            return token if taken else None
        except Exception as exc:
            complain("slots", exc)
    return token if _local_take(key, limit, lease_sec, "local:" + token, at) else None


def give_back(key: str, token: str) -> None:
    with _lock:
        _slots[key].pop("local:" + token, None)
    r = get_redis()
    if r is not None:
        try:
            r.zrem(_PREFIX + "s:" + key, token)
        except Exception as exc:
            complain("slots", exc)


def held(key: str, lease_sec: float) -> int:
    """How many slots are taken right now — for the monitor."""
    at = time.time()
    r = get_redis()
    if r is not None:
        try:
            return int(r.eval(_HELD, 1, _PREFIX + "s:" + key, at, lease_sec))
        except Exception as exc:
            complain("slots", exc)
    with _lock:
        return sum(1 for since in _slots.get(key, {}).values() if at - since < lease_sec)


def _pause(waited: float) -> float:
    # Short at first, so a slot freed a moment later is taken promptly, and
    # jittered, so many waiters do not all ask in the same instant.
    return min(1.0, 0.1 + waited / 10) * random.uniform(0.7, 1.3)


@contextmanager
def slot(key: str, limit: int, lease_sec: float, wait_sec: float = 0.0):
    """Hold one of `limit` slots for the duration of the block (threads)."""
    started = time.monotonic()
    while (token := try_take(key, limit, lease_sec)) is None:
        waited = time.monotonic() - started
        if waited >= wait_sec:
            raise SlotTimeout(key)
        time.sleep(_pause(waited))
    try:
        yield
    finally:
        give_back(key, token)


@asynccontextmanager
async def aslot(key: str, limit: int, lease_sec: float, wait_sec: float = 0.0):
    """Hold one of `limit` slots for the duration of the block (event loop)."""
    started = time.monotonic()
    while (token := try_take(key, limit, lease_sec)) is None:
        waited = time.monotonic() - started
        if waited >= wait_sec:
            raise SlotTimeout(key)
        await asyncio.sleep(_pause(waited))
    try:
        yield
    finally:
        give_back(key, token)
