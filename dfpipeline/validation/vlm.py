"""The AI check: ask a vision model whether the clip is usable — in two ways.

A clip is kept only when both hold:

  1. the task   — the action described really happens, start to finish. The
                  object it names is an example: another object handled the
                  same way is the same action, and a dataset wants that variety;
  2. tracking   — the hand doing it can be followed: from the moment it
                  reaches in until it leaves, the hand (wrist to fingertips) is
                  whole, inside the frame and clear. Every kept clip goes to
                  hand tracking, and a clip the tracker cannot follow is a clip
                  it will discard.

Only clips that survived the free checks get here. The model never receives
video — it receives one timestamped contact sheet, which is what keeps this
affordable. The same answer carries the clip's steps, with their times.
"""

from __future__ import annotations

import logging

from PIL import Image

from dfpipeline import frames as frame_utils
from dfpipeline.llm import LlmClient, LlmUnavailable, _looks_like_transport_error
from dfpipeline.validation.schemas import (
    MODEL_UNREACHABLE,
    ClipVerdict,
    Finding,
    Source,
    Subtask,
    Verdict,
)

log = logging.getLogger("validation.vlm")


SYSTEM = (
    "You decide whether a clip belongs in a robot training dataset. "
    "You are shown ONE image: a grid of frames sampled from a single clip, in time "
    "order, each labelled with its timestamp. Judge only what is visible in them. "
    "A clip is used in two ways, and must be good for both: it shows a robot the "
    "task, and a hand-tracking system follows the operator's hand through it. "
    "The standard is strict: a wrong clip is worse than a missing one — the operator "
    "can simply record it again. A clip passes only when every requirement is clearly "
    "met in every frame; if you are unsure about anything, in any frame, the answer "
    "is no. Never assume what you cannot see. Background clutter and compression "
    "artefacts do not matter; whether the hand — wrist to fingertips — can be seen "
    "clearly while it works does. An arm reaching in from the edge of the picture is "
    "normal, and so are frames before the hand arrives or after it has left. "
    "Reply with JSON only."
)

TEMPLATE = """The operator was asked to perform this task:

Task: {title}
{details}

First, look at EVERY frame in the grid, in order, and say what the hand doing
the task looks like in it. "The hand" is the wrist to the fingertips: the palm
(or the back of the hand) and all five fingers. The forearm and the wrist
crossing the edge of the picture is how an arm reaches in — that is normal and
does not count against the hand.

  "full"     the hand, wrist to fingertips, is inside the picture: every
             fingertip and knuckle and the whole palm are inside it, not hidden
             behind the object, the other hand or the body, and sharp and bright
             enough to make out each finger. The arm running from the wrist to
             the edge of the picture is still "full" — when someone reaches in
             from the camera's side, the arm ALWAYS reaches the edge.
  "partial"  the hand is there, but a FINGERTIP, a knuckle or part of the PALM
             is outside the picture, or hidden, blurred or too dark — or you are
             not sure. The arm or wrist touching the edge is never "partial".
  "absent"   no hand in the picture at all: it has not reached in yet, or has
             already left.

For each one, also say what touches the edge of the picture:
  "nothing"          no part of the hand or arm reaches the edge
  "arm only"         only the forearm or wrist runs off the edge
  "fingers or palm"  a fingertip, knuckle or part of the palm is cut off by it

Then answer each requirement, then give a verdict.

THE TASK
  This is a robot-learning recording: what is learned is the ACTION — the way
  the hands move, grasp, pass, carry and place. The object the task names is an
  example, not a requirement. Another object handled the same way (a block, a
  bottle, a box instead of a cup), another colour, size, table, room or camera
  angle is the same task — and that variety is wanted, not a fault. Judge the
  movements, never the identity of the object.

  task_performed    The action described actually happens, from start to
                    finish, within these frames. Every movement the task asks
                    for (e.g. picked up, passed from one hand to the other,
                    placed in the tray) is there, in that order, done with
                    whatever object the person is holding. A hand near an object, or an
                    object merely present, is not the action. It is false only
                    when the MOVEMENTS differ — a step is missing, done in
                    another way (pushed instead of carried, one hand instead
                    of passing between two), or ends somewhere else (beside
                    the tray instead of in it) — never because the object is
                    not the one named.
  object_visible    The object being handled can be seen while it is handled:
                    you can tell where it is, when it is grasped and where it
                    goes. Any object counts. False only if it is hidden for
                    most of the action or too small or dark to follow.
  view_obstructed   Anything blocks the camera for part of the clip: a hand
                    over the lens, a body in the way, most of the frame covered.

HAND TRACKING — from the moment the hand reaches in until it leaves
  hands_visible     Once the hand has reached in, it stays in view until it
                    leaves: it does not disappear and come back in the middle.
                    Frames before it arrives and after it leaves do not count.
  hand_in_frame     Whenever the hand is in the picture, its palm and every
                    finger are inside it. Check this with "edge" above: only
                    "fingers or palm" crossing the edge makes this false.
                    "arm only" — the forearm or wrist running off the picture —
                    is how every reach looks, and is true.
  hand_clear        Whenever the hand is in the picture it is sharp and bright
                    enough to make out each finger — not a dark shape, not a
                    motion blur.

Then:
  "pass"    every requirement above is met, and you are sure.
  "reject"  any requirement is not met, in any frame.
  "flag"    you cannot tell. Never guess "pass".

Then, only if the verdict is "pass", break the clip into its steps. Use the
timestamps printed on the frames. Two to five steps, in order, each a short
phrase in the imperative — "reach for the cup", "grasp the cup", "lift and
place". The first starts at the first frame's time and the last ends at the
last frame's time, with no gaps or overlaps between them. If you cannot tell
where one step ends and the next begins, give fewer steps rather than guessing
the boundaries.

Reply with exactly this JSON shape:
{{"frames": [{{"t": 0.0, "edge": "nothing" | "arm only" | "fingers or palm",
              "hand": "full" | "partial" | "absent"}}, ...one per frame, in order],
  "verdict": "pass" | "flag" | "reject",
  "confidence": 0.0-1.0,
  "reason": "one short sentence a non-technical operator will understand",
  "checks": {{"task_performed": true|false, "object_visible": true|false,
              "view_obstructed": true|false,
              "hands_visible": true|false, "hand_in_frame": true|false,
              "hand_clear": true|false}},
  "subtasks": [{{"name": "short phrase", "start": 0.0, "end": 2.4}}]}}

The reason must say what you actually saw — "the block is set down beside the
tray, not in it", "the fingers are cut off at the right edge at 2.4s" — rather
than "does not meet the requirements". Name the object actually used ("passes a
black block, then places it in the tray"), and name it the same way in the
steps ("grasp the black block"), so each clip's words match what it shows.
"""

def _for_video(text: str, swaps: list[tuple[str, str]]) -> str:
    for old, new in swaps:
        assert old in text, old  # the grid wording changed: update the swap
        text = text.replace(old, new)
    return text


# The same instructions for a video instead of a grid. Only the words about
# what the model is looking at change; every rule is the grid's own text.
SYSTEM_VIDEO = _for_video(SYSTEM, [
    (("You are shown ONE image: a grid of frames sampled from a single clip, in time "
      "order, each labelled with its timestamp. Judge only what is visible in them. "),
     "You are shown ONE video: the whole clip. Judge only what is visible in it. "),
])

TEMPLATE_VIDEO = _for_video(TEMPLATE, [
    (("First, look at EVERY frame in the grid, in order, and say what the hand doing\n"
      "the task looks like in it."),
     ("The video is {duration} seconds long. First, watch all of it. Then, for EVERY\n"
      "second — t = 0, 1, 2, … up to {last} — say what the hand doing the task looks\n"
      "like at that moment.")),
    ("within these frames.", "within the video."),
    ("Frames before it arrives", "Moments before it arrives"),
    ("any requirement is not met, in any frame.", "any requirement is not met, at any moment."),
    ("Use the\ntimestamps printed on the frames.", "Use the\nvideo's own time, in seconds."),
    (("The first starts at the first frame's time and the last ends at the\n"
      "last frame's time"),
     "The first starts at 0 and the last ends at {duration}"),
    ("...one per frame, in order]", "...one per second, t = 0 to {last}, in order]"),
])


# What each failed requirement tells the operator. Written for the person who
# will record the clip again, so each says what to do differently.
REQUIREMENTS = (
    # (key in "checks", the value that fails it, finding)
    ("task_performed", False, Finding(
        "task_not_performed", "The task described was not carried out in this clip.")),
    ("object_visible", False, Finding(
        "object_not_visible", "The object you are handling is hard to see — keep it in view and in good light.")),
    ("view_obstructed", True, Finding(
        "view_obstructed", "The camera view is blocked during the clip.")),
    ("hands_visible", False, Finding(
        "hand_not_visible",
        "Your hand is hidden for much of the action — keep it in view of the camera.")),
    ("hand_in_frame", False, Finding(
        "hand_out_of_frame",
        "Your hand goes out of the frame while working — keep it inside the picture.")),
    ("hand_clear", False, Finding(
        "hand_unclear",
        "Your hand is too dark or blurred to follow — use more light and move steadily.")),
)


# A pass must be this sure. Below it the model is unsure, and an unsure clip is
# not kept (it is flagged, and a flag is dropped unless keep_unsure_clips).
PASS_MIN_CONFIDENCE = 0.9


def _hand_state(item: dict) -> str:
    """"full", "partial" or "absent" for one frame; anything else is "partial"
    (unsure counts against the clip). The older true/false answer is read too."""
    state = str(item.get("hand", "")).strip().lower()
    # Its own answer about the edge outranks its label: fingers or palm off the
    # picture is a hand the tracker loses, whatever it called it.
    if state == "full" and "finger" in str(item.get("edge", "")).lower():
        return "partial"
    if state in {"full", "partial", "absent"}:
        return state
    if "hand_fully_visible" in item:
        return "full" if item.get("hand_fully_visible") is True else "partial"
    return "partial"


# A hand reaching in from the edge of the picture is cut off for the moment it
# takes to come in, and again as it leaves. Partial frames this close before
# the first whole-hand frame, or after the last, are that — not a cut-off hand
# mid-task. A clip the model itself called "fully visible and sharp" was
# refused for its first and last second.
EDGE_PARTIAL_SEC = 2.0


def _frames_failing(raw) -> list[float] | None:
    """Timestamps of frames where the hand fails the clip.

    The hand may be absent before it first reaches in and after it finally
    leaves, and only partly in the picture for up to EDGE_PARTIAL_SEC while it
    comes in and goes out. Between its first and last whole-hand frames it must
    be whole: a "partial" frame there, or an "absent" one in the middle (it
    vanished mid-action), fails. A clip where the hand is never whole fails at
    every frame it appears in; a clip with no hand at all fails at every frame.

    None when there is no per-frame answer at all — then nothing has been shown
    about the hand, and the clip cannot pass.
    """
    if not isinstance(raw, list) or not raw:
        return None
    states: list[tuple[float, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return None
        try:
            t = round(float(item.get("t", i)), 1)
        except (TypeError, ValueError):
            t = float(i)
        states.append((t, _hand_state(item)))

    present = [i for i, (_, state) in enumerate(states) if state != "absent"]
    if not present:
        return [t for t, _ in states]
    whole = [i for i, (_, state) in enumerate(states) if state == "full"]
    if not whole:
        return [states[i][0] for i in present]
    first, last = present[0], present[-1]
    start, end = states[whole[0]][0], states[whole[-1]][0]

    def fails(i: int, t: float, state: str) -> bool:
        if state == "absent":
            return first < i < last
        if state == "partial":
            coming_in = i < whole[0] and start - t <= EDGE_PARTIAL_SEC
            going_out = i > whole[-1] and t - end <= EDGE_PARTIAL_SEC
            return not (coming_in or going_out)
        return False

    return [t for i, (t, state) in enumerate(states) if fails(i, t, state)]


# A model asked for two to five steps will occasionally give one, or twenty, or
# a step that ends before it starts. None of that should cost the verdict the
# call was really for, so the timeline is taken on its own terms: what parses
# is kept, what does not is dropped.
MAX_SUBTASKS = 8


def _subtasks(raw, video_id: str) -> list[Subtask]:
    """The steps the model named, as far as they make sense."""
    if not isinstance(raw, list):
        return []

    steps: list[Subtask] = []
    for item in raw[:MAX_SUBTASKS]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        # A step that ends before it begins describes nothing, and a zero-length
        # one is a boundary rather than a step.
        if not (end > start >= 0):
            continue
        steps.append(Subtask(name=name[:120], start=start, end=end))

    steps.sort(key=lambda s: s.start)
    if steps and len(steps) != len(raw or []):
        log.info("kept %d of %d steps offered for %s", len(steps), len(raw), video_id)
    return steps


# Said with the task when it is done with both hands. Everything the prompt
# asks of "the hand" then holds for each of them: two hands are tracked, and
# each drives its own arm.
BOTH_HANDS = (
    "This task is done with BOTH hands, and both are tracked. Everywhere below, "
    "\"the hand\" means EACH of the two hands: a frame is \"full\" only when both "
    "hands, each wrist to fingertips, are inside the picture and clear; \"partial\" "
    "when either is cut off at the edge, blurred or too dark; \"absent\" only when "
    "neither hand is in the picture. One hand briefly covering part of the other "
    "while they pass the object between them is part of the task, not \"partial\". "
    "If only one hand ever does the work, task_performed is false."
)


def _task_details(task) -> str:
    bits = []
    if getattr(task, "description", None):
        bits.append(str(task.description))
    steps = getattr(task, "task_flow", None)
    if steps:
        bits.append("Steps: " + "; ".join(str(s) for s in steps[:8]))
    if getattr(task, "hands", 1) == 2:
        bits.append(BOTH_HANDS)
    return "\n".join(bits) or "(no further description provided)"


def verdict_for_sheet(client: LlmClient, sheet: Image.Image, task, video_id: str) -> ClipVerdict:
    """One clip, one image, one call."""
    prompt = TEMPLATE.format(title=getattr(task, "title", "unknown task"), details=_task_details(task))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": frame_utils.to_data_url(sheet)}},
        ]},
    ]
    data = client.chat_json(
        messages,
        # The per-frame answers come first, then the verdict: room for both.
        max_tokens=800,
        # A gate has to be reproducible: the same clip must get the same answer,
        # or a re-check becomes a coin toss. Measured on a borderline clip,
        # temperature 0.1 flipped pass/flag between runs; 0 was stable at every
        # sheet size tried.
        temperature=0.0,
    )
    return _verdict_from_reply(data, video_id, Source.VLM)


def verdict_for_video(client: LlmClient, clip: bytes, duration: float | None, task,
                      video_id: str) -> ClipVerdict:
    """One clip, the clip itself (as small MJPEG), one call.

    The model watches the motion rather than nine stills, and still answers
    for every second, so a failure keeps its time ("at 3s") and the per-moment
    rule below is enforced the same way as for the grid.
    """
    import base64

    seconds = max(1, round(duration or 0))
    prompt = TEMPLATE_VIDEO.format(
        title=getattr(task, "title", "unknown task"), details=_task_details(task),
        duration=f"{duration:.1f}" if duration else "?", last=seconds,
    )
    messages = [
        {"role": "system", "content": SYSTEM_VIDEO},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url",
             "video_url": {"url": "data:video/avi;base64," + base64.b64encode(clip).decode()}},
        ]},
    ]
    # One entry a second, then the verdict and the steps.
    data = client.chat_json(messages, max_tokens=min(4000, 600 + 25 * seconds), temperature=0.0)
    return _verdict_from_reply(data, video_id, Source.VLM)


def _verdict_from_reply(data: dict, video_id: str, source: Source) -> ClipVerdict:
    """What the model said, held to the rules — the same for a grid and a video."""

    raw = str(data.get("verdict", "")).lower().strip()
    verdict = Verdict(raw) if raw in {v.value for v in Verdict} else Verdict.FLAG
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    stated = str(data.get("reason") or "").strip()
    findings: list[Finding] = []
    subtasks = _subtasks(data.get("subtasks"), video_id)

    # A requirement the model says was not met decides the verdict, whatever
    # its summary says: "pass" alongside "the hand leaves the frame" is a clip
    # the tracker will drop, and it must not be kept.
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    failed = [finding for key, fails_on, finding in REQUIREMENTS if checks.get(key) is fails_on]

    # Every frame must show the whole hand. One frame without it is a clip the
    # tracker loses, whatever the summary says — checked here, from the model's
    # own per-frame answers, rather than trusted to its overall impression.
    missing = _frames_failing(data.get("frames"))
    if missing and not any(f.code == "hand_not_visible" for f in failed):
        at = ", ".join(f"{t:g}s" for t in missing[:6])
        failed.append(Finding(
            "hand_not_visible",
            f"Your hand is not fully visible while you work (at {at}) — keep your palm "
            "and all your fingers inside the picture from when you reach in until you let go.",
        ))

    if failed:
        if verdict is Verdict.PASS:
            log.info(
                "vlm said pass for %s while reporting %s — taking the checks",
                video_id,
                ", ".join(f.code for f in failed),
            )
            # It answered the specific question; the fact disqualifies the clip.
            confidence = max(confidence, 0.8)
        verdict = Verdict.REJECT
        findings.extend(failed)
        if stated:
            findings.insert(0, Finding("task_mismatch", stated))
    elif verdict is Verdict.PASS and (missing is None or confidence < PASS_MIN_CONFIDENCE):
        # A pass it cannot show — no per-frame answer — or is not sure of is
        # not a pass. Unsure clips are not kept.
        verdict = Verdict.FLAG
        findings.append(Finding(
            "needs_review",
            "The check could not confirm the hand is fully visible while it works."
            if missing is None
            else f"The check was not sure enough to keep this clip ({confidence:.0%}).",
        ))
    elif verdict is not Verdict.PASS:
        findings.append(
            Finding(
                "task_mismatch" if verdict is Verdict.REJECT else "needs_review",
                stated or "The clip does not clearly show the requested task.",
            )
        )

    return ClipVerdict(
        video_id=video_id,
        verdict=verdict,
        confidence=confidence,
        source=source,
        findings=findings,
        # Only for a clip that will be trained on. A timeline of a clip nobody
        # keeps is a description of something being thrown away.
        subtasks=subtasks if verdict is Verdict.PASS else [],
    )


def safe_verdict(client: LlmClient, task, video_id: str, sheet: Image.Image | None,
                 clip: bytes | None = None, duration: float | None = None) -> ClipVerdict:
    """The check, by video when there is one, by the grid otherwise.

    A model that is down is waited for (UNKNOWN, asked again later) whichever
    way it was asked. A model that answered but would not take the video —
    "could not open video stream" — is asked again with the grid, so a clip is
    never held up by the video route alone.
    """
    if clip is not None:
        try:
            return verdict_for_video(client, clip, duration, task, video_id)
        except LlmUnavailable as exc:
            if _looks_like_transport_error(str(exc)) or sheet is None:
                return _unreachable(video_id, exc)
            log.warning("model would not take %s as a video (%s) — using the frame grid", video_id, exc)
    if sheet is None:
        return ClipVerdict(video_id=video_id, verdict=Verdict.UNKNOWN)
    return safe_verdict_for_sheet(client, sheet, task, video_id)


def _unreachable(video_id: str, exc: Exception) -> ClipVerdict:
    log.warning("vlm unavailable for %s: %s", video_id, exc)
    return ClipVerdict(
        video_id=video_id,
        verdict=Verdict.UNKNOWN,
        source=Source.NONE,
        findings=[Finding(MODEL_UNREACHABLE, f"Model check failed: {exc}")],
    )


def safe_verdict_for_sheet(client: LlmClient, sheet: Image.Image, task, video_id: str) -> ClipVerdict:
    """Never let an AI/gateway failure become a rejection."""
    try:
        return verdict_for_sheet(client, sheet, task, video_id)
    except LlmUnavailable as exc:
        log.warning("vlm unavailable for %s: %s", video_id, exc)
        # Undecided, and the clip says why — and the code tells the job to wait
        # for the model and ask again, rather than settle for no answer.
        return ClipVerdict(
            video_id=video_id,
            verdict=Verdict.UNKNOWN,
            source=Source.NONE,
            findings=[Finding(MODEL_UNREACHABLE, f"Model check failed: {exc}")],
        )
