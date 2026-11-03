"""How far things really are: a depth model on a few frames of the clip.

One phone camera sees no depth. The hand was placed by its size in pixels
against an average hand's size, so a big hand read as near and a small one as
far; the table's distance came from the hand's depth, or 0.5 m when there was
none. Both are guesses, and the robot's reach is built on them.

A monocular metric depth model (MoGe-2, on the GPU) gives every pixel's
distance in metres, from the picture alone. It is run on a dozen frames of the
clip, not all of them: what it is asked for holds for the whole take.

1. **The table.** A plane fitted (RANSAC) through the depth of the frame's
   table area, with the hand masked out; on a phone that sent which way is up
   (lens.up_in_image), only a plane facing up is taken. Its normal and
   distance replace the guessed ones in frame 0, the world frame.
2. **The hand's real size.** Where the hand is seen in those frames, the
   depth at its joints against the depth its reconstruction put them at gives
   one scale for the whole clip. Scaling the 3D hand about the camera keeps it
   exactly on its pixels and moves it only along its lines of sight — the one
   thing a single camera could not tell. A clip whose frames disagree too much
   is left unscaled and flagged.

The model is optional (`RETARGET_DEPTH_MODEL`, default off). Without it, or
on a frame it cannot read, the pipeline does what it did before, and the
bundle says which way each number was made.

The model is loaded once per worker process and kept, on the GPU when there is
one (gpu.py), within its share of the GPU's memory: the DGX's GPU serves the
AI model first.
"""

from __future__ import annotations

import functools
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol

import numpy as np

from dfpipeline.config import settings

log = logging.getLogger("hand2robot.scene")

# Depth sees the skin; a joint is inside the finger. Added to the measured
# depth before it is compared with a joint's.
SKIN_TO_JOINT_M = 0.008
# A clip's hand scale outside this is not a hand of another size, it is a fault.
MIN_SCALE, MAX_SCALE = 0.6, 1.6
# The frames' scales may disagree by this much (interquartile range over the
# median) before the clip's scale is not trusted.
MAX_SCALE_SPREAD = 0.25
# A table plane must hold this share of the sampled points, and lie within
# this angle of up when up is known.
MIN_PLANE_INLIERS = 0.30
MAX_TILT_FROM_UP_DEG = 35.0
PLANE_TOLERANCE_M = 0.012


class DepthModel(Protocol):
    name: str

    def infer(self, rgb: np.ndarray, hfov_deg: float | None) -> tuple[np.ndarray, float | None]:
        """Metric depth (H, W) in metres for one RGB frame, nan where unknown,
        and the horizontal field of view it used (given, or its own estimate)."""


@dataclass
class Plane:
    normal: np.ndarray     # unit, from the camera towards the table (n·X = distance on it),
                           # as camera_reconstruction's and the retargeters' table_normal
    distance: float        # metres from the camera
    inliers: float         # share of the points on it


@dataclass
class Scene:
    model: str
    frames_used: int = 0
    table: Plane | None = None
    up_angle_deg: float | None = None   # between the table's normal and up, where up is known
    hand_scale: float | None = None     # applied to the hand when not None
    scale_measured: float | None = None  # what the frames said, applied or not
    scale_spread: float | None = None
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "depth_model": self.model,
            "depth_frames": self.frames_used,
            "table_source": "depth" if self.table else None,
            "table_inliers": round(self.table.inliers, 3) if self.table else None,
            "table_up_angle_deg": round(self.up_angle_deg, 1) if self.up_angle_deg is not None else None,
            "hand_scale": round(self.hand_scale, 4) if self.hand_scale is not None else None,
            "hand_scale_measured": round(self.scale_measured, 4) if self.scale_measured is not None else None,
            "hand_scale_spread": round(self.scale_spread, 4) if self.scale_spread is not None else None,
            "depth_seconds": round(self.seconds, 1),
            "notes": self.notes,
        }


# ------------------------------------------------------------- the geometry

def backproject(depth: np.ndarray, K: np.ndarray, step: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """3D points (N, 3) in camera axes from a depth map, on a grid every
    `step` pixels, and their (N, 2) pixels. Unknown depth is skipped."""
    h, w = depth.shape
    v, u = np.mgrid[step // 2:h:step, step // 2:w:step]
    z = depth[v, u]
    ok = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    u, v, z = u[ok].astype(float), v[ok].astype(float), z[ok].astype(float)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    return pts, np.stack([u, v], axis=1)


def fit_plane(points: np.ndarray, up: np.ndarray | None = None, iterations: int = 300,
              tolerance: float = PLANE_TOLERANCE_M, seed: int = 0) -> Plane | None:
    """The dominant plane through `points` by RANSAC, refined by least squares
    on its inliers. With `up`, only planes within MAX_TILT_FROM_UP_DEG of it."""
    n = len(points)
    if n < 50:
        return None
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    cos_tilt = math.cos(math.radians(MAX_TILT_FROM_UP_DEG))
    for _ in range(iterations):
        a, b, c = points[rng.choice(n, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if up is not None and abs(float(normal @ up)) < cos_tilt:
            continue
        count = int(np.count_nonzero(np.abs((points - a) @ normal) < tolerance))
        if count > best_count:
            best, best_count = (normal, a), count
    if best is None or best_count / n < MIN_PLANE_INLIERS:
        return None
    normal, a = best
    inliers = points[np.abs((points - a) @ normal) < tolerance]
    centre = inliers.mean(axis=0)
    # The refined normal: the direction the inliers vary least in.
    normal = np.linalg.svd(inliers - centre, full_matrices=False)[2][-1]
    distance = float(normal @ centre)
    if distance < 0:              # point the normal from the camera to the table
        normal, distance = -normal, -distance
    if up is not None and abs(float(normal @ up)) < cos_tilt:
        return None
    share = float(np.count_nonzero(np.abs(points @ normal - distance) < tolerance)) / n
    return Plane(normal=normal.astype(float), distance=abs(distance), inliers=share)


def depth_at(depth: np.ndarray, uv: np.ndarray, radius: int = 2) -> np.ndarray:
    """The depth at each pixel (median of a small window: an edge pixel can
    read the table behind a finger), nan where unknown or off the frame."""
    h, w = depth.shape
    out = np.full(len(uv), np.nan)
    for i, (u, v) in enumerate(np.round(uv).astype(int)):
        if not (0 <= u < w and 0 <= v < h):
            continue
        patch = depth[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1]
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size:
            out[i] = float(np.median(patch))
    return out


def hand_scale(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, int] | None:
    """One scale for the hand from (measured depth, reconstructed depth) per
    sampled frame: each frame's median ratio, then the median over frames.
    Returns (scale, spread, frames) or None when too few frames saw the hand."""
    per_frame = []
    for measured, reconstructed in pairs:
        ok = np.isfinite(measured) & np.isfinite(reconstructed) & (reconstructed > 0.05)
        if np.count_nonzero(ok) < 6:      # most of the hand hidden: not a measurement
            continue
        per_frame.append(float(np.median((measured[ok] + SKIN_TO_JOINT_M) / reconstructed[ok])))
    if len(per_frame) < 2:
        return None
    ratios = np.asarray(per_frame)
    centre = float(np.median(ratios))
    q75, q25 = np.percentile(ratios, [75, 25])
    return centre, float((q75 - q25) / centre), len(per_frame)


def table_points(depth: np.ndarray, K: np.ndarray, hand_boxes: list[list[float]]) -> np.ndarray:
    """Points of the frame that can be the table: below the top third (a phone
    looking down at a table sees it there), outside the hands."""
    pts, uv = backproject(depth, K)
    keep = uv[:, 1] > depth.shape[0] / 3.0
    for x1, y1, x2, y2 in hand_boxes:
        keep &= ~((uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2))
    return pts[keep]


# ------------------------------------------------------------- the measuring

def sample_frames(total: int, count: int) -> list[int]:
    """`count` frames spread over the clip, always including frame 0 (the
    world frame)."""
    if total <= 0:
        return []
    count = max(1, min(count, total))
    return sorted({0, *np.linspace(0, total - 1, count).round().astype(int).tolist()})


def measure(video: Path, K: np.ndarray, hfov_deg: float | None, total_frames: int,
            hands: dict[int, list[tuple[np.ndarray, np.ndarray]]],
            boxes: dict[int, list[list[float]]], up: np.ndarray | None,
            model: DepthModel | None = None) -> Scene | None:
    """Measure the table and the hand's scale for one clip.

    `hands`: frame -> [(2D keypoints (21, 2), reconstructed joints in camera
    axes (21, 3))] for every tracked hand in that frame; `boxes`: frame -> the
    hands' boxes, kept out of the table. None when no depth model is set.
    """
    model = model or depth_model()
    if model is None:
        return None
    import cv2

    started = time.perf_counter()
    scene = Scene(model=model.name)
    wanted = sample_frames(total_frames, settings.retarget_depth_frames)
    cap = cv2.VideoCapture(str(video))
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                continue
            try:
                depth, _ = model.infer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hfov_deg)
            except Exception as exc:  # noqa: BLE001 - one unreadable frame is not the clip
                scene.notes.append(f"frame {idx}: {str(exc)[:120]}")
                continue
            scene.frames_used += 1
            if idx == 0 and scene.table is None:
                scene.table = fit_plane(table_points(depth, K, boxes.get(0, [])), up)
                if scene.table is None:
                    scene.notes.append("no table plane found in frame 0")
            for keypoints, joints in hands.get(idx, []):
                pairs.append((depth_at(depth, keypoints), np.asarray(joints, dtype=float)[:, 2]))
    finally:
        cap.release()

    if scene.table is not None and up is not None:
        scene.up_angle_deg = math.degrees(math.acos(min(1.0, abs(float(scene.table.normal @ up)))))

    found = hand_scale(pairs)
    if found is None:
        scene.notes.append("the hand was not seen in enough sampled frames to measure its size")
    else:
        scale, spread, _ = found
        scene.scale_measured, scene.scale_spread = scale, spread
        if not MIN_SCALE <= scale <= MAX_SCALE:
            scene.notes.append(f"hand scale {scale:.2f} is outside {MIN_SCALE}-{MAX_SCALE}: not applied")
        elif spread > MAX_SCALE_SPREAD:
            scene.notes.append(f"the frames disagree on the hand's size (spread {spread:.2f}): not applied")
        else:
            scene.hand_scale = scale
    scene.seconds = time.perf_counter() - started
    log.info("scene: %s", scene.summary())
    return scene


def estimate_hfov(video: Path, total_frames: int, frames: int = 3,
                  model: DepthModel | None = None) -> float | None:
    """The lens's horizontal field of view, by the depth model, when the phone
    did not say: the median over a few frames. None without a model."""
    model = model or depth_model()
    if model is None:
        return None
    import cv2

    found = []
    cap = cv2.VideoCapture(str(video))
    try:
        for idx in sample_frames(total_frames, frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                continue
            try:
                _, hfov = model.infer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None)
            except Exception as exc:  # noqa: BLE001 - no estimate is the guess, not a failure
                log.warning("field of view not estimated on frame %d: %s", idx, exc)
                continue
            if hfov:
                found.append(hfov)
    finally:
        cap.release()
    return float(np.median(found)) if found else None


# ------------------------------------------------------------- the models

@functools.cache
def depth_model() -> DepthModel | None:
    """The configured depth model, loaded once per process; None when off or
    when it cannot load (the clip is then measured the old way, and the
    log says why)."""
    name = (settings.retarget_depth_model or "off").strip().lower()
    if name in ("", "off", "none"):
        return None
    try:
        if name.startswith("moge"):
            return MoGe(name)
    except Exception as exc:  # noqa: BLE001 - no model is a fallback, not a failed clip
        log.warning("depth model %s could not load (%s); measuring without it", name, exc)
        return None
    log.warning("unknown depth model %r; measuring without it", name)
    return None


class MoGe:
    """Microsoft's MoGe-2 (MIT): metric point maps from one image, plain
    PyTorch. `fov_x` is given when the phone told us the lens (lens.py), so
    the model spends nothing on guessing it."""

    #: RETARGET_DEPTH_MODEL -> Hugging Face checkpoint (MoGe's README table).
    CHECKPOINTS: ClassVar[dict[str, str]] = {
        "moge-2-vits": "Ruicheng/moge-2-vits-normal",   # 35M parameters
        "moge-2-vitb": "Ruicheng/moge-2-vitb-normal",   # 104M
        "moge-2-vitl": "Ruicheng/moge-2-vitl-normal",   # 331M
    }

    def __init__(self, name: str) -> None:
        import torch
        from moge.model.v2 import MoGeModel

        from dfpipeline import gpu

        if name not in self.CHECKPOINTS:
            raise ValueError(f"one of {sorted(self.CHECKPOINTS)}")
        self.name = name
        self.device = gpu.torch_device()
        self.torch = torch
        # Baked into the image (scripts/download_perception_weights.py); the
        # Hub only where it was not, as on a laptop or Colab.
        local = Path(settings.retarget_models_dir) / "moge" / name / "model.pt"
        source = str(local) if local.is_file() else self.CHECKPOINTS[name]
        self.model = MoGeModel.from_pretrained(source).to(self.device).eval()
        log.info("depth model %s on %s", name, self.device)

    def infer(self, rgb: np.ndarray, hfov_deg: float | None) -> tuple[np.ndarray, float | None]:
        torch = self.torch
        image = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        out = self.model.infer(image, fov_x=hfov_deg, resolution_level=settings.retarget_depth_resolution,
                               use_fp16=self.device != "cpu")
        depth = out["depth"].float().cpu().numpy()
        mask = out.get("mask")
        if mask is not None:
            depth = np.where(mask.cpu().numpy(), depth, np.nan)
        used = hfov_deg
        intrinsics = out.get("intrinsics")
        if used is None and intrinsics is not None:
            # MoGe's intrinsics are normalised: fx in widths of the image.
            fx = float(intrinsics[0, 0])
            used = math.degrees(2.0 * math.atan(0.5 / fx)) if fx > 0 else None
        return depth, used
