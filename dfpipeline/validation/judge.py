"""The clip check: look at each clip, run the free rules, ask the model.

Runs on the pipeline (queue `check`), between the app's
`ValidationService.prepare_*`, which says where each clip is, and its `save_*`,
which writes what was reached. No database here. The verdicts travel back as
plain JSON (`verdict_to_wire`), and exactly what was reached here is what the
app writes.

Guarantees:
  * an infrastructure failure is never a rejection (UNKNOWN, goes to a human)
  * a rejection always carries a reason the operator can read
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

from dfpipeline import frames as frame_utils
from dfpipeline import sources
from dfpipeline import video as video_utils
from dfpipeline.config import settings
from dfpipeline.llm import LlmClient
from dfpipeline.validation import checks as rules
from dfpipeline.validation.schemas import (
    ClipFacts,
    ClipVerdict,
    Source,
    Thresholds,
    Verdict,
    verdict_to_wire,
)
from dfpipeline.validation.vlm import safe_verdict

log = logging.getLogger("validation")

# Fetching a clip back from storage is the slowest and least reliable part of
# a check. A name that will not resolve says nothing about the clip.
INSPECT_ATTEMPTS = 3
INSPECT_BACKOFF_SEC = 2.0


def _inspect_and_rule(url: str, size: int | None, video_id: str, t: Thresholds,
                      label: str | None, headers: dict | None):
    """Blocking part: fetch frames, measure, run the free rules. Runs in a thread."""
    probe, frames, sheet = frame_utils.inspect_clip(
        url, size, t.frames_per_clip, t.frame_width, headers=headers, label=label
    )
    luma, motion = rules.measure(frames)
    facts = ClipFacts(
        video_id=video_id, duration=probe.duration, width=probe.width, height=probe.height,
        size_bytes=size, frame_count=len(frames), mean_luma=luma, motion=motion,
        decodable=bool(frames),
    )
    return facts, rules.run_checks(facts, t), sheet


def _inspect_with_retries(url: str, size: int | None, video_id: str, t: Thresholds,
                          label: str | None, headers: dict | None):
    """Fetch and read a clip, giving a flaky link a second chance.

    A clip is fetched back from storage to be looked at, and a name that will
    not resolve or a connection that drops is not evidence about the
    recording. Giving up after one try marked the clip undecided — and with
    unattended decisions on, undecided is kept and paid. A network blip must
    not become a payment for something nobody ever saw.
    """
    last: Exception | None = None
    for attempt in range(1, INSPECT_ATTEMPTS + 1):
        try:
            return _inspect_and_rule(url, size, video_id, t, label, headers)
        except Exception as exc:  # noqa: BLE001 - a flaky fetch is retried, then undecided
            last = exc
            if attempt == INSPECT_ATTEMPTS:
                break
            wait = INSPECT_BACKOFF_SEC * attempt
            log.warning(
                "could not reach %s (attempt %d/%d): %s — retrying in %.1fs",
                video_id, attempt, INSPECT_ATTEMPTS, exc, wait,
            )
            time.sleep(wait)
    raise last  # type: ignore[misc]


def judge_clip(clip: dict, task, t: Thresholds, llm: LlmClient) -> ClipVerdict:
    """Everything the check does to one clip: look, rule, then ask.

    No database: the clip arrives as a description of where to read it
    (`app/ml/sources.py`), so this runs wherever the work is sent.
    """
    video_id = clip["video_id"]
    label = f"task: {task.title}" if task else None
    try:
        url, headers = sources.resolve(clip)
        facts, findings, sheet = _inspect_with_retries(
            url, clip.get("size_bytes"), video_id, t, label, headers
        )
    except Exception as exc:  # noqa: BLE001 - fetch/decode blew up: undecided, not rejected
        log.warning("could not inspect %s after retrying: %s", video_id, exc)
        return ClipVerdict(video_id=video_id, verdict=Verdict.UNKNOWN)

    if findings:  # tier 1 said no - never pay for a model call
        return ClipVerdict(
            video_id=video_id, verdict=Verdict.REJECT, confidence=1.0,
            source=Source.RULES, findings=findings, facts=facts,
        )

    if sheet is None or not llm.configured:
        return ClipVerdict(video_id=video_id, verdict=Verdict.UNKNOWN, facts=facts)

    clip_video = _clip_for_model(url, headers, facts.duration, video_id)
    verdict = safe_verdict(llm, task, video_id, sheet, clip_video, facts.duration)
    verdict.facts = facts
    return verdict


def _clip_for_model(url: str, headers: dict | None, duration: float | None,
                    video_id: str) -> bytes | None:
    """The small MJPEG copy the model watches, or None to use the frame grid."""
    if not settings.check_with_video:
        return None
    try:
        return video_utils.clip_for_model(
            url, settings.check_video_width, settings.check_video_fps,
            settings.check_video_max_frames, duration, headers,
        )
    except Exception as exc:  # noqa: BLE001 - the grid still works without it
        log.warning("could not make a video of %s for the model (%s) — using the frame grid", video_id, exc)
        return None


def judge_clips(spec: dict) -> dict:
    """The compute half of a check: every clip in `spec`, one after another.

    `spec` is built by `ValidationService.prepare_*` and holds everything the
    check needs — where each clip is, the task's words, the thresholds — so
    this needs no database.
    """
    t = Thresholds(**spec["thresholds"])
    task = SimpleNamespace(**spec["task"]) if spec.get("task") else None
    llm = LlmClient.from_settings()
    return {
        "verdicts": [verdict_to_wire(judge_clip(clip, task, t, llm)) for clip in spec["clips"]]
    }
