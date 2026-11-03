"""The one Redis connection, or none.

Redis is optional. With `REDIS_URL` empty the app runs exactly as a single
machine with no Redis: jobs are drained by the built-in poller and every limit
is counted inside its own process. With it set, Celery carries the jobs and the
limits become true across every process and machine — which is the point of
having it, since four API workers each counting on their own let four times the
configured number through.

Callers never assume Redis is reachable. Anything that asks it a question is
written to fall back to its local answer when the question fails, because a
Redis restart must not take uploads or logins down with it.
"""

from __future__ import annotations

import logging
import threading
import time

from dfpipeline.config import settings

log = logging.getLogger(__name__)

_client = None
_override = None
_lock = threading.Lock()

# When Redis last failed, so a dead Redis is reported once a minute rather
# than once per request.
_last_complaint = 0.0


def get_redis():
    """The shared client, or None when no Redis is configured."""
    global _client
    if _override is not None:
        return _override
    if not settings.redis_url:
        return None
    if _client is None:
        with _lock:
            if _client is None:
                import redis

                # Short timeouts: every caller has a local fallback, and a slow
                # answer is worse than no answer for a request that is waiting.
                # The pool notices a fork by itself, so prefork workers are safe.
                _client = redis.Redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0,
                    health_check_interval=30,
                )
    return _client


def use_redis(client) -> None:
    """Swap the client — tests pass a fakeredis instance, or None to go back."""
    global _override
    _override = client


def complain(what: str, exc: Exception) -> None:
    """Log a Redis failure, at most once a minute."""
    global _last_complaint
    at = time.monotonic()
    if at - _last_complaint > 60:
        _last_complaint = at
        log.warning("redis unavailable for %s, counting locally instead: %s", what, exc)
