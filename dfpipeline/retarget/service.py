"""One clip -> one SO-101 trajectory, and the decision whether it is worth training on.

The pipeline itself is vendored (see `__init__.py`); this is the part that is
ours. It runs a clip through, reads what came out, and applies the gate.

The gate is the point. A clip the check passed is a clip where a human did the
task — which says nothing about whether the hand and the object were tracked
well enough to drive an arm. On the POC's own material the same code produced,
from two videos that both exited zero:

    6.49.56   object in 98% of frames, 1.95x slowdown
    6.49.16   object in 14% of frames, 4.7x slowdown

The second is an arm thrashing around an object it never found. Nothing
downstream can tell them apart, so this does.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from dfpipeline.config import settings

log = logging.getLogger("retarget.clip")



class RetargetFailed(RuntimeError):
    """The pipeline could not produce a trajectory at all. Distinct from the
    gate refusing one it did produce: this is worth retrying, that is not."""


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_hand_metrics(out_dir: Path) -> dict:
    """What the hand stage produced, before any robot is involved.

    These are facts about the person and the camera — was the hand visible,
    was the object found — and they are the same whatever arm the clip is
    later exported for.
    """
    import json as _json

    bundle = _json.loads((out_dir / "hand_stage.json").read_text(encoding="utf-8"))

    def one_hand(folder: Path) -> dict:
        coverage = seconds = share = None
        hand_file = folder / "trajectory.json"
        if hand_file.is_file():
            meta = (_json.loads(hand_file.read_text(encoding="utf-8")).get("metadata") or {})
            coverage = hand_coverage(meta)
            seconds, share = hand_span(meta, bundle.get("fps"))
        return {"hand_coverage": coverage, "hand_seconds": seconds, "hand_share": share}

    per_hand = None
    if bundle.get("hands"):
        # Both hands: each measured on its own; the headline is the weaker one,
        # and a hand never followed counts as nothing.
        per_hand = {}
        for side in bundle["hands"]:
            m = one_hand(out_dir / side) if (out_dir / side).is_dir() else \
                {"hand_coverage": 0.0, "hand_seconds": 0.0, "hand_share": 0.0}
            per_hand[side] = m
        headline = {k: min((m[k] for m in per_hand.values() if m[k] is not None), default=None)
                    for k in ("hand_coverage", "hand_seconds", "hand_share")}
    else:
        headline = one_hand(out_dir)
    coverage, seconds, share = headline["hand_coverage"], headline["hand_seconds"], headline["hand_share"]

    object_fraction = None
    object_file = out_dir / "object_trajectory.json"
    if object_file.is_file():
        frames = _json.loads(object_file.read_text(encoding="utf-8")).get("frames") or []
        if frames:
            object_fraction = round(sum(1 for f in frames if f.get("detected")) / len(frames), 4)

    return {
        "hand_coverage": coverage,
        "hand_seconds": seconds,
        "hand_share": share,
        "object_frames_fraction": object_fraction,
        "frames": bundle.get("total_frames"),
        "fps": bundle.get("fps"),
        "object_label": (bundle.get("object") or {}).get("label"),
        # How the distances were made: phone lens or guess, depth model or not
        # (pipeline._perception). Absent from bundles made before.
        **({"perception": bundle["perception"]} if bundle.get("perception") else {}),
        **({"hands": per_hand} if per_hand is not None else {}),
    }


def hand_span(meta: dict, fps: float | None) -> tuple[float | None, float | None]:
    """How long the hand was followed — first to last reconstructed frame — in
    seconds, and as a share of the clip. (None, None) for older bundles."""
    first, last = _f(meta.get("hand_first_frame")), _f(meta.get("hand_last_frame"))
    total, rate = _f(meta.get("total_frames")), _f(fps or meta.get("fps"))
    if first is None or last is None or not total or not rate:
        return None, None
    frames = last - first + 1
    return round(frames / rate, 2), round(frames / total, 4)


def hand_coverage(meta: dict) -> float | None:
    """How much of the time the hand was there, it was reconstructed.

    Over the span from the first frame the hand was reconstructed in to the
    last — not the whole clip. A reach starts with the hand outside the
    picture and ends with it gone; counting those seconds as missing hand
    failed every ordinary pick-and-place, and the AI check already allows them.
    Older bundles without the span are measured over the whole clip.
    """
    done = _f(meta.get("reconstructed_frames"))
    if done is None:
        return None
    first, last = _f(meta.get("hand_first_frame")), _f(meta.get("hand_last_frame"))
    span = (last - first + 1) if first is not None and last is not None else _f(meta.get("total_frames"))
    if not span:
        return None
    return round(min(1.0, done / span), 4)


def hand_gate(metrics: dict) -> str | None:
    """Whether these coordinates are worth keeping, before any arm is solved.

    Both bars here are about the recording, not the robot: if the hand was
    never seen, or the object never found, no amount of retargeting will help
    and every arm would fail the same way. Catching it here means the operator
    hears about the framing after one clip rather than after fifty.

    A two-hand clip needs each hand to clear the hand bars on its own: an arm
    whose hand was half seen is half an episode.
    """
    s = settings
    hands = metrics.get("hands")
    checks = [(f"The {side} hand", m) for side, m in hands.items()] if hands else [("The hand", metrics)]
    for who, m in checks:
        reason = _hand_reason(who, m)
        if reason:
            return reason

    objects = metrics.get("object_frames_fraction")
    if objects is not None and objects < s.retarget_min_object_frames:
        return (
            f"The object was found in only {objects:.0%} of frames "
            f"(needs {s.retarget_min_object_frames:.0%}), so the grasp timing cannot be trusted."
        )
    return None


def _hand_reason(who: str, metrics: dict) -> str | None:
    """Why one hand's tracking is not good enough, or None."""
    s = settings
    seconds, share = metrics.get("hand_seconds"), metrics.get("hand_share")
    if seconds is not None and share is not None and (
        seconds < s.retarget_min_hand_seconds or share < s.retarget_min_hand_share
    ):
        return (
            f"{who} was followed for only {seconds:.1f}s, {share:.0%} of the clip "
            f"(needs {s.retarget_min_hand_seconds:.1f}s and {s.retarget_min_hand_share:.0%}), "
            "so the episode would be a fragment of the task. Keep the whole hand in the frame, "
            "close to the camera, from the start of the task to the end."
        )

    coverage = metrics.get("hand_coverage")
    if coverage is not None and coverage < s.retarget_min_hand_coverage:
        return (
            f"{who} was reconstructed in only {coverage:.0%} of frames "
            f"while it was in view (needs {s.retarget_min_hand_coverage:.0%}). Keep the whole hand, "
            "wrist included, inside the frame and clearly visible while you work."
        )
    return None


def read_metrics(out_dir: Path) -> dict:
    """Everything the gate and the review screens want, from what the run wrote."""
    written = out_dir / "robot_trajectory.json"   # an imported robot's (retarget_urdf.py)
    if not written.is_file():
        written = out_dir / "so101_trajectory.json"
    robot = json.loads(written.read_text(encoding="utf-8"))
    records = robot.get("trajectory") or []
    if not records:
        raise RetargetFailed("the retargeter produced no frames")

    retiming = robot.get("retiming") or {}
    action = [r for r in records if r.get("phase") == "action"]

    # The hand: what the world-space fusion actually reconstructed, against the
    # length of the clip. `trajectory.json` is always written when we get here
    # (for two arms, one per hand in its folder; the weaker hand counts).
    coverages = []
    for folder in [out_dir / side for side in robot.get("arms") or []] or [out_dir]:
        hand_file = folder / "trajectory.json"
        if hand_file.is_file():
            meta = (json.loads(hand_file.read_text(encoding="utf-8")).get("metadata") or {})
            coverages.append(hand_coverage(meta))
    known = [c for c in coverages if c is not None]
    coverage = min(known) if known else None

    # The object: the weakest stage in the pipeline, and the one that decides
    # whether the grasp schedule means anything.
    object_fraction = None
    object_file = out_dir / "object_trajectory.json"
    if object_file.is_file():
        frames = json.loads(object_file.read_text(encoding="utf-8")).get("frames") or []
        if frames:
            object_fraction = round(sum(1 for f in frames if f.get("detected")) / len(frames), 4)

    errors = sorted(e for r in action if (e := _f(r.get("position_error_m"))) is not None)
    median_error = errors[len(errors) // 2] if errors else None

    return {
        "hand_coverage": coverage,
        "object_frames_fraction": object_fraction,
        "reachable_fraction": _f(retiming.get("hand_path_reachable_fraction")),
        "slowdown": _f(retiming.get("slowdown")),
        "peak_limit_ratio": _f(retiming.get("peak_speed_or_accel_vs_limit")),
        "position_error_median_m": round(median_error, 5) if median_error is not None else None,
        "sample_count": len(action),
        "calibrated": bool((robot.get("lerobot_calibration") or {}).get("file_found")),
    }


def gate(metrics: dict) -> str | None:
    """Why this trajectory is not fit to train on, or None if it is.

    Every threshold is a setting. These numbers came from three test videos;
    they will move once there is a corpus behind them, and a constant would
    mean a deploy to move one.
    """
    # The hand bars are not repeated here. They belong to the clip rather than
    # to any arm, `hand_gate` holds them, and nothing reaches this point
    # without having cleared them.
    s = settings
    reachable = metrics.get("reachable_fraction")
    if reachable is not None and reachable < s.retarget_min_reachable:
        return (
            f"Only {reachable:.0%} of the hand path is inside the arm's reach "
            f"(needs {s.retarget_min_reachable:.0%}). The work happened where this robot cannot go."
        )

    error = metrics.get("position_error_median_m")
    if error is not None and error > s.retarget_max_position_error_m:
        return (
            f"The arm stays {100 * error:.1f} cm from the hand's path (median; limit "
            f"{100 * s.retarget_max_position_error_m:.0f} cm): it cannot follow this motion. "
            "Most often the hand was only partly tracked, so the path jumps."
        )

    slowdown = metrics.get("slowdown")
    if slowdown is not None and slowdown > s.retarget_max_slowdown:
        return (
            f"The robot needs {slowdown:.1f}x the human's time (limit {s.retarget_max_slowdown:.1f}x). "
            "That much retiming means the solver was fighting the joint limits, not following a path."
        )

    peak = metrics.get("peak_limit_ratio")
    if peak is not None and peak > s.retarget_max_peak_limit:
        return f"A joint exceeds its speed or acceleration limit ({peak:.2f}x). This is a pipeline fault, not a bad clip."

    return None


def _require_models() -> None:
    from dfpipeline.retarget import paths

    absent = paths.missing()
    if absent:
        raise RetargetFailed(
            "The retargeting models are not installed: "
            + ", ".join(absent)
            + ". Run scripts/download_retarget_weights.py."
        )


def track_hand(video_path: Path, work_dir: Path, hands: int = 1,
               camera: dict | None = None) -> tuple[dict, Path]:
    """Reconstruct the hand and the object. No robot involved.

    Runs in a worker thread — a minute of CPU, mostly MediaPipe over every
    frame twice. What it writes is the same answer for every arm the clip is
    ever exported for, so it is done once and kept.
    """
    from dfpipeline.retarget.pipeline import run_hand_stage

    _require_models()
    out_dir = work_dir / "hand"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    code = run_hand_stage(video_path, out_dir, hands=hands, camera=camera)
    if code != 0 or not (out_dir / "hand_stage.json").is_file():
        raise RetargetFailed(
            "No hand could be reconstructed from this clip. Most often the hand is "
            "outside the frame, or cropped at its edge, so the detector never acquires it."
        )

    metrics = read_hand_metrics(out_dir)
    metrics["duration_sec"] = round(time.perf_counter() - started, 1)
    return metrics, out_dir


def retarget_urdf_bundle(hand_dir: Path, robot_spec: dict, work_dir: Path) -> tuple[dict, Path]:
    """Solve an imported robot's arm (its URDF and profile) for a hand
    trajectory that is already reconstructed — one hand, or both onto its two
    sides. The same outputs as the SO-101's, under robot_trajectory.json."""
    import numpy as np

    from dfpipeline.embodiment.convert import robot_from_spec
    from dfpipeline.retarget import grasp
    from dfpipeline.retarget.pipeline import _load_hand_frames
    from dfpipeline.retarget.retarget_urdf import retarget_to_urdf

    out_dir = work_dir / "robot"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    robot = robot_from_spec(robot_spec, work_dir, "robot")
    bundle = json.loads((hand_dir / "hand_stage.json").read_text(encoding="utf-8"))
    table_normal = np.asarray(bundle["table_normal"], dtype=np.float64) if bundle.get("table_normal") else None
    if bundle.get("hands"):
        sides = [s for s in bundle["hands"] if (hand_dir / s).is_dir()]
        if len(sides) != 2:
            raise RetargetFailed("A two-arm trajectory needs both hands, and one was never reconstructed.")
        hands = {s: _load_hand_frames(hand_dir / s) for s in sides}
        grips = {s: grasp.read_grip(hand_dir / s) for s in sides}
    else:
        hands = {"": _load_hand_frames(hand_dir)}
        grips = {"": grasp.read_grip(hand_dir)}
    if not all(hands.values()):
        raise RetargetFailed("The hand trajectory has no frames to retarget.")
    try:
        trajectory = retarget_to_urdf(robot, hands, table_normal, grips, out_dir, robot.name,
                                      bundle.get("table_distance_m"))
    except Exception as exc:  # noqa: BLE001 - reported as no trajectory, as for the SO-101
        log.error("%s retargeting failed: %s", robot.name, exc)
        raise RetargetFailed(f"The arm could not be solved for this hand path: {exc}") from None

    for name in ("trajectory.json", "object_trajectory.json", "left/trajectory.json", "right/trajectory.json"):
        source = hand_dir / name
        if source.is_file() and not (out_dir / name).exists():
            (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, out_dir / name)
    metrics = read_metrics(out_dir)
    metrics["duration_sec"] = round(time.perf_counter() - started, 1)
    return metrics, trajectory


def retarget_bundle(hand_dir: Path, calibration: dict | None, work_dir: Path) -> tuple[dict, Path]:
    """Solve one arm for a hand trajectory that is already reconstructed.

    Seconds rather than a minute, because the expensive half is already done.
    The calibration is written to this run's own directory and bound with a
    context variable rather than a module constant, because several of these
    share a worker process.
    """
    from dfpipeline.retarget import paths
    from dfpipeline.retarget.pipeline import run_robot_stage

    _require_models()
    out_dir = work_dir / "robot"
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration_file = None
    if calibration:
        calibration_file = work_dir / "lerobot_calibration.json"
        calibration_file.write_text(json.dumps(calibration), encoding="utf-8")

    started = time.perf_counter()
    with paths.use_calibration(calibration_file):
        code = run_robot_stage(hand_dir, out_dir)

    trajectory = out_dir / "so101_trajectory.json"
    if code != 0 or not trajectory.is_file():
        raise RetargetFailed(
            "The arm could not be solved for this hand path. The coordinates are fine; "
            "the trajectory is not."
        )

    # The hand exports sit beside the robot one so the metrics read as before.
    for name in ("trajectory.json", "object_trajectory.json", "left/trajectory.json", "right/trajectory.json"):
        source = hand_dir / name
        if source.is_file() and not (out_dir / name).exists():
            (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, out_dir / name)

    metrics = read_metrics(out_dir)
    metrics["duration_sec"] = round(time.perf_counter() - started, 1)
    return metrics, trajectory


def bundle_path(video_id: str) -> str:
    """Where a clip's hand coordinates live. One per clip, not one per robot."""
    return f"tracks/{settings.retarget_pipeline_version}/{video_id}.zip"


def store_bundle(hand_dir: Path, path: str) -> None:
    """Put the hand stage's output in the media repo, as one archive.

    Four files travel together — the readable trajectory, the array form, the
    object, and the handover the arm needs — and a robot export wants all of
    them. One object is one upload and one download rather than four.
    """
    import zipfile

    from dfpipeline.storage import get_storage

    archive = hand_dir.parent / "hand-bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for item in sorted(hand_dir.rglob("*")):
            # The tracked view is stored separately so a player can stream it;
            # inside the archive too would be the same video twice. A two-hand
            # clip's hands are in their own folders, kept as they are.
            if item.is_file() and item.suffix != ".mp4":
                zf.write(item, item.relative_to(hand_dir).as_posix())
    with archive.open("rb") as fh:
        get_storage().put(path, fh)


def fetch_bundle(path: str, work_dir: Path) -> Path:
    """Unpack a stored hand bundle so the robot stage can read it."""
    import zipfile

    from dfpipeline.storage import get_storage

    local = get_storage().fetch_local(path)
    out = work_dir / "hand"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local) as zf:
        zf.extractall(out)
    return out


def store_path(robot_id: str, video_id: str) -> str:
    return f"retargets/{settings.retarget_pipeline_version}/{robot_id}/{video_id}.json.gz"


def store(trajectory: Path, path: str) -> None:
    """Put the trajectory in the media repo, gzipped.

    It is one record per robot sample with a chain point per link — megabytes
    of it, and about nine tenths air.
    """
    from dfpipeline.storage import get_storage

    packed = trajectory.with_suffix(".json.gz")
    with trajectory.open("rb") as src, gzip.open(packed, "wb", compresslevel=6) as out:
        shutil.copyfileobj(src, out)
    with packed.open("rb") as fh:
        get_storage().put(path, fh)


def load(path: str) -> dict:
    """Read a stored trajectory back, for the export build."""
    from dfpipeline.storage import get_storage

    local = get_storage().fetch_local(path)
    with gzip.open(local, "rt", encoding="utf-8") as fh:
        return json.load(fh)
