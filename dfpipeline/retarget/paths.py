"""Where the retargeting pipeline's model files live.

Vendored from Hand2Robot, which ran from its own repository root and could
write `Path("models/hand_landmarker.task")` and be right. Here the working
directory is the backend, the worker may be started from anywhere, and the
calibration file is not one file at all — it belongs to whichever robot the
export chose.

So: the fixed assets resolve against one configured directory, and the
calibration is a context variable rather than a constant. Four retarget jobs
run at once in a single worker process, and a module-level path set per job
would hand one robot's calibration to another's clip.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from dfpipeline.config import settings

_calibration: ContextVar[Path | None] = ContextVar("retarget_calibration", default=None)


def models_dir() -> Path:
    return Path(settings.retarget_models_dir).resolve()


def hand_landmarker() -> Path:
    return models_dir() / "hand_landmarker.task"


def object_detector() -> Path:
    return models_dir() / "efficientdet_lite0.tflite"


def urdf() -> Path:
    return models_dir() / "so101" / "so101_new_calib.urdf"


def calibration() -> Path:
    """This run's LeRobot calibration file.

    Absent, the retargeter assumes symmetric joint ranges and reports
    `file_found: false` — a trajectory that is fine to train on and wrong to
    send to a particular arm. A path that does not exist is returned as-is;
    the pipeline already treats a missing file that way.
    """
    chosen = _calibration.get()
    return chosen if chosen is not None else models_dir() / "so101" / "lerobot_calibration.json"


@contextmanager
def use_calibration(path: Path | str | None):
    token = _calibration.set(Path(path) if path else None)
    try:
        yield
    finally:
        _calibration.reset(token)


def missing() -> list[str]:
    """The fixed assets that are not on disk. Checked before a job starts."""
    return [str(p) for p in (hand_landmarker(), object_detector(), urdf()) if not p.exists()]
