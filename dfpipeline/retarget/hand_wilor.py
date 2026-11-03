"""WiLoR: the hand detector, a 3D hand model on the GPU.

WiLoR (CVPR 2025) fits a full MANO hand per frame, with its own hand
detector: about 15 mm average joint error, and it holds up when part of the
hand is hidden — which, holding things, it always is. On the Colab A/B/C
test it replaced MediaPipe: 5x less jitter, and a hand path that stays
within reach of the table instead of jumping a metre in depth.

It gives `HandDetection`s, so tracking, reconstruction, fusion and the robot
stage are unchanged. Its 3D joints (metres, MANO: wrist, then thumb, index,
middle, ring, little, base to tip) are fitted to its 2D joints with the
clip's real lens (lens.py) by `HandReconstructor`.

Through WiLoR-mini (`wilor_mini`), the packaged inference of it. Licences:
WiLoR's weights are CC BY-NC-ND 4.0 and MANO's are non-commercial; the
detector is Ultralytics YOLO. Research only (docs/HAND2ROBOT_2.md).

Every frame is read; the model is loaded once per worker process and kept,
on the GPU within its share of memory (gpu.torch_device). There is no CPU
path: without a GPU the clip fails.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np

from dfpipeline.retarget.hand_detection import HandDetection

log = logging.getLogger("hand2robot.hand_wilor")

#: The detector's bar for a hand; WiLoR-mini's default. Low: false hands are dropped by the trackers and the PnP fit.
MIN_CONFIDENCE = 0.3


@functools.cache
def _pipeline():
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    from dfpipeline import gpu

    device = gpu.torch_device()     # "cuda", or it raises
    dtype = torch.float16
    log.info("WiLoR on %s (%s)", device, dtype)
    # Baked into the image (scripts/download_perception_weights.py); else
    # WiLoR-mini fetches them into this folder on first use.
    from dfpipeline.config import settings

    weights = Path(settings.retarget_models_dir) / "wilor"
    return WiLorHandPose3dEstimationPipeline(device=torch.device(device), dtype=dtype, verbose=False,
                                             wilor_pretrained_dir=str(weights))


def to_detection(found: dict, width: int, height: int) -> HandDetection | None:
    """One of WiLoR-mini's hands as a HandDetection, or None if incomplete."""
    preds = found.get("wilor_preds") or {}
    try:
        kp2 = np.asarray(preds["pred_keypoints_2d"], dtype=np.float32).reshape(21, 2)
        kp3 = np.asarray(preds["pred_keypoints_3d"], dtype=np.float32).reshape(21, 3)
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(kp2).all() and np.isfinite(kp3).all()):
        return None
    x1, y1, x2, y2 = (float(v) for v in found.get("hand_bbox") or kp2_box(kp2))
    return HandDetection(
        hand_side="right" if float(found.get("is_right", 1.0)) >= 0.5 else "left",
        bbox=[max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)],
        confidence=float(found.get("confidence", 0.9)),
        track_id=1,
        keypoints_2d=kp2,
        landmarks_3d_norm=None,
        # Root-relative, metres: PnP needs the shape, not where it is.
        landmarks_world=kp3 - kp3[0],
    )


def kp2_box(kp2: np.ndarray) -> list[float]:
    lo, hi = kp2.min(axis=0), kp2.max(axis=0)
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]


class WilorDetector:
    """Hands in every frame of a video, by WiLoR."""

    name = "wilor"

    def __init__(self) -> None:
        # WiLoR's own camera (its focal length and translation) is not used:
        # only its joints are, placed with the clip's real lens by
        # HandReconstructor. So nothing of the lens is passed in here.
        _pipeline()          # load now: a missing model fails here, not mid-clip

    def detect_video(self, video_path: str | Path, fps: float, progress_callback=None,
                     hands: int = 1) -> list[list[HandDetection]]:
        import cv2

        pipe = _pipeline()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out: list[list[HandDetection]] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                try:
                    found = pipe.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), hand_conf=MIN_CONFIDENCE)
                except Exception as exc:  # noqa: BLE001 - one bad frame is a missing frame
                    log.warning("WiLoR failed on frame %d: %s", len(out), exc)
                    found = []
                frame_hands = [d for d in (to_detection(f, w, h) for f in found) if d is not None]
                out.append(frame_hands[: max(1, hands) + 1])
                if progress_callback and len(out) % 20 == 0 and total:
                    progress_callback(5.0 + 45.0 * len(out) / total, "Stage 1/3: hands (WiLoR)...")
        finally:
            cap.release()
        return out


def make_detector() -> WilorDetector:
    """The hand detector: WiLoR, the only one."""
    return WilorDetector()
