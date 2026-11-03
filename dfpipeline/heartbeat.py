"""The pipeline's hours, from the app, and its "here" back to the app.

Hours: set on the app's Settings page and kept in Redis (`pipeline:hours`);
read here every tick, so a change there reaches this server within 30 s, with
no redeploy. PIPELINE_HOURS/PIPELINE_TZ are the fallback when Redis has none.

Heartbeat: outside its hours the pipeline runs no workers, so nothing answers
the app and a server that is down looks exactly like one asleep. So the
scheduler writes `pipeline:heartbeat:<node>` every tick, asleep or not, and it
expires after three missed ticks: the Monitor reads it (CONTRACT.md).
"""

from __future__ import annotations

import json
import os
import time

from dfpipeline import PROTOCOL_VERSION, hours
from dfpipeline.config import settings

HOURS_KEY = "pipeline:hours"
KEY = "pipeline:heartbeat:{node}"
TTL_SEC = 90


def node() -> str:
    return settings.node_name or os.environ.get("HOSTNAME") or "pipeline"


def working_hours() -> tuple[str, str]:
    """(hours, tz) as set in the app — "" hours is always — else this server's env."""
    from dfpipeline.redis import complain, get_redis

    fallback = (settings.pipeline_hours or "", settings.pipeline_tz)
    client = get_redis()
    if client is None:
        return fallback
    try:
        raw = client.get(HOURS_KEY)
    except Exception as exc:  # noqa: BLE001 - Redis away: the env's hours
        complain("pipeline hours", exc)
        return fallback
    if not raw:
        return fallback
    try:
        data = json.loads(raw)
    except ValueError:
        return fallback
    return data.get("hours") or "", data.get("tz") or settings.pipeline_tz


def beat(asleep: bool, window: str, tz: str) -> None:
    """Write this server's "here". Never raises: a missed beat is just a later one."""
    from dfpipeline import gpu
    from dfpipeline.redis import complain, get_redis

    client = get_redis()
    if client is None:
        return
    payload = {
        "node": node(), "at": time.time(), "asleep": asleep, "hours": window or None, "tz": tz,
        "opens_in_sec": hours.seconds_until_open(window, tz), "protocol": PROTOCOL_VERSION,
        "gpu": gpu.nvidia_present() and gpu.wanted(),
    }
    try:
        client.set(KEY.format(node=node()), json.dumps(payload), ex=TTL_SEC)
    except Exception as exc:  # noqa: BLE001 - Redis away: tried again next tick
        complain("heartbeat", exc)
