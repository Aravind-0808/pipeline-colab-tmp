"""The heavy work the pipeline does, by name — during its hours only.

The app server prepares each job, sends `pipeline.compute(name, spec, protocol)`,
and writes what comes back. The names, the specs and the results are the
contract between the two projects (CONTRACT.md):

    validate.judge   the AI check: frames, free rules, the model's two-part
                     verdict and the clip's timed steps            (queue check)
    track.hand       reconstruct the hand and the object; stage the bundle
    retarget.solve   solve the arm from a hand bundle; stage the trajectory
    export.build     build a LeRobot dataset into the store (MinIO)      (queue build)
    dataset.publish  send a stored dataset version to Hugging Face     (queue build)

Everything a compute needs is in its spec, and everything it decides is in
its result. Nothing here has a database to ask.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dfpipeline import PROTOCOL_VERSION
from dfpipeline.config import settings

log = logging.getLogger("pipeline.compute")

Compute = Callable[[dict], Any]
_COMPUTE: dict[str, Compute] = {}


def compute(name: str) -> Callable[[Compute], Compute]:
    def wrap(fn: Compute) -> Compute:
        _COMPUTE[name] = fn
        return fn

    return wrap


def run(name: str, spec: dict, protocol: int | None = None) -> dict:
    """Run a compute and wrap the outcome so it can cross the queue.

    An exception does not travel well between machines, so it becomes its
    message here; the app server raises it again, where the job's retry rules apply.
    A message from a different version of the contract is refused outright —
    two servers on two releases is a deploy mistake, and a clear error is
    better than a result written wrongly.
    """
    if protocol != PROTOCOL_VERSION:
        return {
            "ok": False,
            "error": (
                f"protocol mismatch: the app server sent v{protocol}, this pipeline speaks "
                f"v{PROTOCOL_VERSION}. Deploy the same release to both servers."
            ),
        }
    fn = _COMPUTE.get(name)
    if fn is None:
        return {"ok": False, "error": f"No compute function '{name}'"}
    try:
        return {"ok": True, "value": fn(spec)}
    except Exception as exc:
        log.exception("compute %s failed", name)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}


# ------------------------------------------------------------------ helpers


def plain(value):
    """`value` as JSON — numpy numbers and paths included — to cross a queue."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def is_missing(exc: Exception) -> bool:
    """A file that is not there (yet) - as opposed to one that would not open."""
    return isinstance(exc, FileNotFoundError) or type(exc).__name__.endswith("EntryNotFoundError")


# ------------------------------------------------------------- validate.judge


@compute("validate.judge")
def judge(spec: dict) -> dict:
    """The AI check of one or more clips. See dfpipeline/validation/judge.py."""
    from dfpipeline.validation.judge import judge_clips

    return judge_clips(spec)


# ----------------------------------------------------------------- track.hand


@compute("track.hand")
def track_hand(spec: dict) -> dict:
    """The hand over the clip (MediaPipe or WiLoR, with the depth model where
    set); the hand bundle stored.

    The gate is decided here, with this server's thresholds: whether the hand
    and the object were seen well enough to be worth solving an arm against.
    """
    from dfpipeline import sources
    from dfpipeline.retarget import service as rs
    from dfpipeline.retarget.service import RetargetFailed

    try:
        with tempfile.TemporaryDirectory(prefix="track-") as tmp:
            work = Path(tmp)
            source = sources.materialize(spec["clip"], work)
            # One hand unless the task says both (an older app sends no "hands").
            # And what the phone said about its camera (lens.py): absent from
            # an older app, and from a phone that could not say.
            camera = spec.get("camera") if isinstance(spec.get("camera"), dict) else None
            metrics, hand_dir = rs.track_hand(source, work, hands=int(spec.get("hands") or 1),
                                              camera=camera)
            rs.store_bundle(hand_dir, spec["bundle_path"])
    except RetargetFailed as exc:
        # The pipeline could not read this clip. An answer about the clip,
        # not a crash: recorded, and not retried.
        return {"failed": str(exc)[:1000]}
    metrics = plain(metrics)
    return {
        "metrics": metrics,
        "bundle_path": spec["bundle_path"],
        "reason": rs.hand_gate(metrics),
    }


# ------------------------------------------------------------- retarget.solve


@compute("retarget.solve")
def retarget_solve(spec: dict) -> dict:
    """Solve the arm from the stored hand bundle; the trajectory staged for the Hub."""
    from dfpipeline.retarget import service as rs
    from dfpipeline.retarget.service import RetargetFailed

    try:
        with tempfile.TemporaryDirectory(prefix="retarget-") as tmp:
            work = Path(tmp)
            try:
                hand_dir = rs.fetch_bundle(spec["bundle_path"], work)
            except Exception as exc:
                if is_missing(exc):
                    # Made on another machine and not committed yet.
                    return {"bundle_pending": True}
                raise
            robot = spec.get("robot")
            if robot and robot.get("package_path"):
                # An imported robot: its own arm, from its URDF.
                metrics, trajectory = rs.retarget_urdf_bundle(hand_dir, robot, work)
            else:
                metrics, trajectory = rs.retarget_bundle(hand_dir, spec.get("calibration"), work)
            rs.store(trajectory, spec["store_path"])
    except RetargetFailed as exc:
        return {"failed": str(exc)[:1000]}
    metrics = plain(metrics)
    return {"metrics": metrics, "trajectory_path": spec["store_path"], "reason": rs.gate(metrics)}


# --------------------------------------------------------------- export.build


def _progress_reporter(export_id: str):
    """Send a build's progress to the app server, at most every few seconds.

    It travels as a small message to the app server's "db" queue, which writes
    it; a build reports after every episode, and the app's database is asked
    for short writes.
    """
    last = {"at": 0.0, "step": None}

    def report(fraction: float, step: str) -> None:
        try:
            at = time.monotonic()
            if step == last["step"] and at - last["at"] < 5:
                return
            last.update(at=at, step=step)
            from dfpipeline.worker import send_progress

            send_progress(export_id, fraction, step)
        except Exception as exc:  # never allowed to fail the build
            log.debug("could not report export progress (%s): %s", step, exc)

    return report


@compute("export.build")
def build_dataset(spec: dict) -> dict:
    """Build the dataset into the store (MinIO). Publishing is separate."""
    from dfpipeline.lerobot_builder import build_to_store

    payload = spec["payload"]

    # Trajectories may have been made on another machine and reach the Hub
    # only with its next commit. A build cannot proceed without them, so it
    # says so and the app server tries again shortly.
    if payload.get("robot_id"):
        from dfpipeline.storage import get_storage

        storage = get_storage()
        missing = 0
        for ep in payload["episodes"]:
            try:
                storage.fetch_local(ep["trajectory_path"])
            except Exception as exc:
                if not is_missing(exc):
                    raise
                missing += 1
        if missing:
            return {"pending": missing}

    root = Path(settings.export_data_dir) / payload["repo_id"].replace("/", "__")
    result = build_to_store(payload, root, _progress_reporter(payload["export_id"]))

    # The build is in the store now, so the copy here is a duplicate.
    if not settings.export_keep_local:
        # Where it was actually built, which is not always where it was asked
        # to build — a locked leftover sends it next door.
        built_at = Path(result.get("root") or root)
        if discard_build(built_at):
            log.info("cleared the local build at %s", built_at)
    return plain(result)


@compute("dataset.publish")
def publish_dataset(spec: dict) -> dict:
    """Send a stored dataset version to its Hugging Face repo — the only thing
    that ever goes to Hugging Face, and only when an admin asks."""
    from dfpipeline.lerobot_builder import publish

    return plain(publish(spec["store_prefix"], spec["repo_id"], spec.get("private", True),
                         _progress_reporter(spec["export_id"]), version=spec.get("version")))


@compute("dataset.convert")
def convert_dataset(spec: dict) -> dict:
    """A finished dataset on the Hub, converted to another robot into a new
    repo — the videos copied on the Hub, never downloaded (dfpipeline/embodiment)."""
    from dfpipeline.embodiment.convert import convert_on_hub

    return plain(convert_on_hub(spec, _progress_reporter(spec["export_id"])))


def discard_build(root: Path) -> bool:
    """Delete a build directory, never at the cost of the job that made it."""
    if not root.exists():
        return False
    try:
        shutil.rmtree(root)
        if root.name == "dataset" and "-build-" in root.parent.name:
            # Only remove the empty workspace allocated for this build.
            try:
                root.parent.rmdir()
            except OSError:
                pass
        return True
    except OSError as exc:
        log.warning("could not clear the local build at %s: %s", root, exc)
        return False


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def sweep_stale_builds(data_dir: str | Path | None = None) -> dict:
    """Delete export workspaces that can never be used again.

    A build is reused only when it carries `.export-complete.json`, so one
    without it — left by a restart mid-build — is dead by definition. Run at
    worker start, where by definition nothing is building.
    """
    root = Path(data_dir or settings.export_data_dir)
    if not root.is_dir():
        return {"removed": 0, "freed_bytes": 0}

    removed, freed = 0, 0
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        dataset = entry / "dataset" if (entry / "dataset").is_dir() else entry
        if (dataset / ".export-complete.json").is_file():
            continue
        size = _dir_size(entry)
        if discard_build(entry):
            removed += 1
            freed += size
            log.info("swept unfinished export build %s (%.1f MB)", entry.name, size / 1e6)
    return {"removed": removed, "freed_bytes": freed}
