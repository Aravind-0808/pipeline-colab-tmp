"""Hand back short jobs a stopped worker took and never finished.

A task is acknowledged when it finishes, so a worker that dies holding one
leaves it in Redis's "unacked" set — and Redis gives it back only after the
visibility timeout, nine hours, set that long so a dataset build is never
handed to a second worker while the first is still at it. A clip check left
behind by a worker stopped with two Ctrl+C sat as "running" all day.

Checks and hand tracking take minutes, so for those queues a worker looks for
messages taken a while ago that no live worker is running or holding, and puts
them back. Builds and publishing keep the long timeout.
"""

from __future__ import annotations

import json
import logging
import threading
import time

log = logging.getLogger("pipeline.recovery")

# Kombu's names for the messages a worker has taken and not yet acknowledged.
UNACKED = "unacked"
UNACKED_INDEX = "unacked_index"
LOCK = "pipeline:hand-back"

SHORT_QUEUES = ("check", "track")
# Taken longer ago than this. A worker that has only just taken a message may
# not be answering yet; nothing in these queues is new at two minutes.
MIN_AGE_SEC = 120
EVERY_SEC = 600


def hand_back(redis, running: set[str], queues=SHORT_QUEUES, min_age_sec: float = MIN_AGE_SEC,
              now: float | None = None) -> list[str]:
    """Put abandoned messages from `queues` back at the front of their queue.

    `running` holds the task ids every live worker reports as active or
    reserved. Returns the ids handed back.
    """
    at = time.time() if now is None else now
    handed: list[str] = []
    for tag, taken in redis.zrange(UNACKED_INDEX, 0, -1, withscores=True):
        if at - taken < min_age_sec:
            continue
        raw = redis.hget(UNACKED, tag)
        if not raw:
            continue
        try:
            payload, _exchange, routing_key = json.loads(raw)
            task_id = payload["headers"]["id"]
        except (ValueError, KeyError, TypeError):
            continue
        if routing_key not in queues or task_id in running:
            continue
        pipe = redis.pipeline()
        # The right end is where workers take from: it runs next.
        pipe.rpush(routing_key, json.dumps(payload))
        pipe.hdel(UNACKED, tag)
        pipe.zrem(UNACKED_INDEX, tag)
        pipe.execute()
        handed.append(task_id)
    return handed


def _running_task_ids(app) -> set[str] | None:
    """Every task a live worker is running or holding; None if they could not
    be asked, in which case nothing is handed back."""
    try:
        inspect = app.control.inspect(timeout=3.0)
        ids: set[str] = set()
        for answer in (inspect.active(), inspect.reserved()):
            if answer is None:
                continue
            for tasks in answer.values():
                ids.update(t.get("id") for t in tasks if t.get("id"))
        return ids
    except Exception:
        log.exception("could not ask the workers what they are running")
        return None


def sweep_once(app, redis) -> list[str]:
    if not redis.set(LOCK, "1", nx=True, ex=60):
        return []  # another worker is sweeping
    try:
        running = _running_task_ids(app)
        if running is None:
            return []
        handed = hand_back(redis, running)
        if handed:
            log.warning("handed back %d job(s) a stopped worker left unfinished: %s",
                        len(handed), ", ".join(handed))
        return handed
    finally:
        redis.delete(LOCK)


def start(app) -> None:
    """Sweep shortly after start, then every ten minutes, off the main thread."""
    from dfpipeline.redis import get_redis

    def loop() -> None:
        time.sleep(10)
        while True:
            try:
                redis = get_redis()
                if redis is not None:
                    sweep_once(app, redis)
            except Exception:
                log.exception("hand-back sweep failed; trying again later")
            time.sleep(EVERY_SEC)

    threading.Thread(target=loop, name="hand-back", daemon=True).start()
