"""Shared vocabulary for clip validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Verdict(StrEnum):
    PASS = "pass"
    FLAG = "flag"
    REJECT = "reject"
    UNKNOWN = "unknown"  # we could not decide (e.g. gateway down) - never punish for this


# Which requirement a finding belongs to, so a reviewer — and the operator —
# can see whether the task or the framing was the problem.
TASK_FINDINGS = ("object_not_visible", "view_obstructed", "task_not_performed", "task_mismatch")
TRACKING_FINDINGS = ("hand_not_visible", "hand_out_of_frame", "hand_unclear")
# No answer because the model could not be reached: the check waits for it and
# asks again, never settles for it.
MODEL_UNREACHABLE = "model_unreachable"


class Source(StrEnum):
    RULES = "rules"
    VLM = "vlm"
    NONE = "none"


@dataclass(frozen=True)
class Finding:
    """One concrete, explainable reason. `detail` is shown to the operator."""

    code: str
    detail: str

    def as_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail}


@dataclass
class ClipFacts:
    """What the cheap pass measured about a clip — the input to the rules."""

    video_id: str
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    frame_count: int = 0
    mean_luma: float | None = None       # 0-255, low = black screen
    motion: float | None = None          # mean abs frame-to-frame difference
    decodable: bool = True


@dataclass
class Subtask:
    """One step of the demonstration, and when it happens.

    This is what LeRobot stores per episode: a name and a span. It comes from
    the check rather than a second pass over the video, because the check is
    already looking at a timestamped grid of the clip and one answer can carry
    both.
    """

    name: str
    start: float
    end: float

    def as_dict(self) -> dict:
        return {"name": self.name, "start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass
class ClipVerdict:
    video_id: str
    verdict: Verdict = Verdict.UNKNOWN
    confidence: float = 0.0
    source: Source = Source.NONE
    findings: list[Finding] = field(default_factory=list)
    # What the probe measured. Kept at full precision: the stored column is a
    # whole number, and rounding 1.5s up to 2s would contradict the reason the
    # reviewer is reading next to it.
    facts: ClipFacts | None = None
    # The steps the model saw, in order. Empty when it was not asked, could not
    # tell, or the clip was rejected — a clip nobody will train on does not need
    # a timeline.
    subtasks: list[Subtask] = field(default_factory=list)

    @property
    def reason(self) -> str:
        """A single human-readable sentence for the operator."""
        return " ".join(f.detail for f in self.findings) or "No reason recorded."

    def as_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "verdict": str(self.verdict),
            "confidence": round(self.confidence, 3),
            "subtasks": [s.as_dict() for s in self.subtasks],
            "source": str(self.source),
            "findings": [f.as_dict() for f in self.findings],
            **(
                {
                    "duration": self.facts.duration,
                    "width": self.facts.width,
                    "height": self.facts.height,
                    "size_bytes": self.facts.size_bytes,
                }
                if self.facts
                else {}
            ),
        }


@dataclass
class Thresholds:
    """Admin-tunable knobs, loaded from platform settings."""

    min_duration: float = 3.0
    max_duration: float = 900.0
    min_width: int = 240
    black_luma: float = 12.0
    min_motion: float = 1.5
    frames_per_clip: int = 9
    frame_width: int = 320
    auto_reject_confidence: float = 0.8
    shadow_mode: bool = True
    enabled: bool = True
    # Decide the clips the model was not sure about, instead of parking them
    # for a person. Without it a flagged or unreadable clip holds its whole
    # recording open forever if nobody reviews.
    decide_unsure: bool = False
    # A clip the model looked at and could not vouch for. Kept only if the
    # platform says a maybe is good enough, which for a training set it is not.
    keep_unsure_clips: bool = False


# ------------------------------------------------------------------- wire
#
# A verdict reached on one machine is written on another, so it crosses a
# queue as plain JSON. `as_dict` above is the shape the API shows and drops
# measurements; these keep everything, so the verdict written is exactly the
# verdict reached.


def verdict_to_wire(clip: ClipVerdict) -> dict:
    return {
        "video_id": clip.video_id,
        "verdict": str(clip.verdict),
        "confidence": clip.confidence,
        "source": str(clip.source),
        "findings": [f.as_dict() for f in clip.findings],
        "subtasks": [{"name": s.name, "start": s.start, "end": s.end} for s in clip.subtasks],
        "facts": (
            {
                "video_id": clip.facts.video_id,
                "duration": clip.facts.duration,
                "width": clip.facts.width,
                "height": clip.facts.height,
                "size_bytes": clip.facts.size_bytes,
                "frame_count": clip.facts.frame_count,
                "mean_luma": clip.facts.mean_luma,
                "motion": clip.facts.motion,
                "decodable": clip.facts.decodable,
            }
            if clip.facts
            else None
        ),
    }


def verdict_from_wire(data: dict) -> ClipVerdict:
    facts = data.get("facts")
    return ClipVerdict(
        video_id=data["video_id"],
        verdict=Verdict(data["verdict"]),
        confidence=float(data.get("confidence") or 0.0),
        source=Source(data.get("source") or Source.NONE),
        findings=[Finding(f["code"], f["detail"]) for f in data.get("findings") or []],
        subtasks=[
            Subtask(name=s["name"], start=float(s["start"]), end=float(s["end"]))
            for s in data.get("subtasks") or []
        ],
        facts=ClipFacts(**facts) if facts else None,
    )


def factors(findings: list[dict] | None, verdict: str | None) -> dict[str, str | None]:
    """The two requirements, each "pass", "fail" or None (not judged), read
    off a clip's verdict and findings — for the review screens."""
    if verdict == "pass":
        return {"task": "pass", "tracking": "pass"}
    if verdict != "reject":
        return {"task": None, "tracking": None}
    codes = {f.get("code") for f in findings or [] if isinstance(f, dict)}
    task_failed = bool(codes & {"task_not_performed", "object_not_visible", "view_obstructed"})
    tracking_failed = bool(codes & set(TRACKING_FINDINGS))
    if not (task_failed or tracking_failed):
        # Rejected by the free rules (black, frozen, too short), or with only
        # the model's own sentence: the clip as a whole failed the task.
        return {"task": "fail", "tracking": None}
    return {
        "task": "fail" if task_failed else "pass",
        "tracking": "fail" if tracking_failed else "pass",
    }
