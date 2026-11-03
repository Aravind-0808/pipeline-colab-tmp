"""When the pipeline works: one daily window, e.g. "22:00-08:00".

The DGX is the pipeline's only at night. Its workers run inside the window and
are stopped outside it (dfpipeline/scheduler.py); the app holds work back
outside it rather than send it to a queue nobody is reading, and its Monitor
says the pipeline is asleep rather than down.

A window may cross midnight. Empty, or one that cannot be read, means always.
Times are in the zone given (PIPELINE_TZ), so the window does not move with the
server's clock.

The same module exists in the app (app/core/hours.py): keep them identical.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def parse(hours: str | None) -> tuple[time, time] | None:
    """("22:00-08:00") -> (22:00, 08:00); None for "always"."""
    if not hours or "-" not in hours:
        return None
    try:
        start, end = (time.fromisoformat(part.strip()) for part in hours.split("-", 1))
    except ValueError:
        return None
    return None if start == end else (start, end)


# Short names people write, for the IANA keys zoneinfo knows. "IST" alone is
# not a key — it is ambiguous (India, Ireland, Israel) — so it would crash.
ZONE_ALIASES = {"IST": "Asia/Kolkata"}


def _zone(tz: str) -> ZoneInfo:
    key = (tz or "").strip()
    return ZoneInfo(ZONE_ALIASES.get(key.upper(), key))


def _local(at: datetime | None, tz: str) -> datetime:
    at = at or datetime.now(timezone.utc)
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(_zone(tz))


def is_open(hours: str | None, tz: str, at: datetime | None = None) -> bool:
    """Whether the pipeline works at `at` (now when omitted)."""
    window = parse(hours)
    if window is None:
        return True
    start, end = window
    now = _local(at, tz).time()
    if start < end:
        return start <= now < end
    return now >= start or now < end  # crosses midnight


def _next(at_local: datetime, when: time) -> datetime:
    candidate = at_local.replace(hour=when.hour, minute=when.minute, second=0, microsecond=0)
    if candidate <= at_local:
        candidate += timedelta(days=1)
    return candidate


def seconds_until_open(hours: str | None, tz: str, at: datetime | None = None) -> int:
    """0 when open; otherwise how long until the window opens."""
    if is_open(hours, tz, at):
        return 0
    start, _ = parse(hours)  # type: ignore[misc]  (not open, so there is a window)
    local = _local(at, tz)
    return max(1, int((_next(local, start) - local).total_seconds()))


def seconds_until_close(hours: str | None, tz: str, at: datetime | None = None) -> int | None:
    """How long the window stays open from now; None when always open, 0 when closed."""
    window = parse(hours)
    if window is None:
        return None
    if not is_open(hours, tz, at):
        return 0
    local = _local(at, tz)
    return max(1, int((_next(local, window[1]) - local).total_seconds()))
