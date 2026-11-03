"""Camera and World Reconstruction Module (lightweight, plane-based visual odometry).

Hand2Robot stage 3 (camera / world):
Estimates per-frame 6-DoF camera poses in world coordinates (T_world_camera [4, 4])
and camera intrinsics K [3, 3] across the video sequence.

Capture setup: a handheld phone looking down at a work surface while the other
hand performs the task. The dominant background is the (planar) table, so camera
motion is recovered from the table's homography instead of the essential matrix.
With small handheld motion over a plane the essential matrix is degenerate
(measured: >85% of frames failed), while the homography is well-conditioned
and, given the camera-to-table distance, yields METRIC translation.

Method:
1. Track background features (hand regions masked, forward-backward LK check).
2. Frame-to-frame homography (RANSAC), chained to H_0->i (frame 0 = world frame).
3. Decompose H_0->i = R + t n^T / d with K; keep the physically valid solution whose
   plane normal matches the reference normal; scale t by the table distance d.

`static_camera=True` skips odometry entirely (tripod / stand recordings).

Outputs per frame:
- T_world_camera: 4x4 rigid transformation matrix from camera frame to world frame
- camera_intrinsics: 3x3 projection matrix
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("hand2robot.camera_reconstruction")

DEFAULT_PLANE_DISTANCE_M = 0.5  # camera-to-table distance if it cannot be estimated
MIN_TRACKS = 20
HAND_MASK_PAD = 0.25         # enlarge hand bbox by 25% per side when masking


@dataclass
class CameraTrajectoryResult:
    """Estimated camera trajectory across all video frames."""

    poses_world_camera: np.ndarray  # Shape: (T, 4, 4)
    intrinsics: np.ndarray          # Shape: (3, 3) or (T, 3, 3)
    source: str                     # "plane-visual-odometry" or "static-camera"
    table_normal: np.ndarray | None = None  # table plane normal in the frame-0 camera frame


def _hand_mask(shape: tuple[int, int], bboxes: list[list[float]] | None) -> np.ndarray:
    """255 = usable background, 0 = hand regions."""
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    for bbox in bboxes or []:
        x1, y1, x2, y2 = bbox
        px, py = (x2 - x1) * HAND_MASK_PAD, (y2 - y1) * HAND_MASK_PAD
        x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
        mask[y1:y2, x1:x2] = 0
    return mask


class CameraPoseEstimator:
    """Plane-based monocular visual odometry for a handheld camera over a table."""

    def estimate_trajectory(
        self,
        video_path: str | Path,
        hand_bboxes: dict[int, list[list[float]]] | None = None,
        static_camera: bool = False,
        plane_distance_m: float | None = None,
        K: np.ndarray | None = None,
    ) -> CameraTrajectoryResult:
        """Estimate 6-DoF camera trajectory T_world_camera for each frame in the video.

        Args:
            video_path: Path to input MP4 video.
            hand_bboxes: frame_idx -> hand bboxes, masked out of feature tracking.
            static_camera: If True, the camera is assumed fixed (identity poses).
            plane_distance_m: Camera-to-table distance at frame 0 (sets metric scale).
            K: The lens (lens.py); None for the guess used before the phone said.

        Returns:
            CameraTrajectoryResult containing (T, 4, 4) world-from-camera transformation matrices.
        """
        hand_bboxes = hand_bboxes or {}
        d0 = float(plane_distance_m or DEFAULT_PLANE_DISTANCE_M)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video for camera reconstruction: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)

        if K is None:
            from dfpipeline.retarget.lens import guess

            K = guess(width, height).K

        if static_camera:
            n = 0
            while cap.grab():
                n += 1
            cap.release()
            log.info("Static camera mode: using identity camera poses for %d frames.", n)
            return CameraTrajectoryResult(
                poses_world_camera=np.repeat(np.eye(4, dtype=np.float32)[None], max(n, 1), axis=0),
                intrinsics=K,
                source="static-camera",
            )

        log.info(
            "Computing camera trajectory for %d frames (%dx%d), table distance %.2f m...",
            total_frames, width, height, d0,
        )

        feature_params = {"maxCorners": 400, "qualityLevel": 0.01, "minDistance": 8, "blockSize": 7}
        lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }

        K64 = K.astype(np.float64)
        H_0i = np.eye(3)                 # maps frame-0 pixels to frame-i pixels
        ref_normal: np.ndarray | None = None
        last_pose = np.eye(4, dtype=np.float32)
        poses: list[np.ndarray] = []
        prev_gray = None
        n_failed = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                H_step = self._frame_homography(
                    prev_gray, gray,
                    _hand_mask(prev_gray.shape, hand_bboxes.get(frame_idx - 1)),
                    _hand_mask(gray.shape, hand_bboxes.get(frame_idx)),
                    feature_params, lk_params,
                )
                if H_step is None:
                    n_failed += 1  # keep previous pose (assume no motion this frame)
                else:
                    H_0i = H_step @ H_0i
                    H_0i /= H_0i[2, 2]
                    pose, ref_normal = self._pose_from_homography(H_0i, K64, d0, ref_normal)
                    if pose is not None:
                        last_pose = pose
                    else:
                        n_failed += 1

            poses.append(last_pose.copy())
            prev_gray = gray
            frame_idx += 1

        cap.release()
        poses_arr = np.stack(poses, axis=0) if poses else np.eye(4, dtype=np.float32)[np.newaxis, ...]

        # Light smoothing of camera translation (removes residual estimation noise;
        # real handheld shake is low-frequency enough to survive a short window).
        if len(poses_arr) >= 7:
            try:
                from scipy.signal import savgol_filter
                cam_t = poses_arr[:, :3, 3]
                win = min(7, len(cam_t) if len(cam_t) % 2 != 0 else len(cam_t) - 1)
                poses_arr[:, :3, 3] = savgol_filter(cam_t, window_length=win, polyorder=2, axis=0)
            except ImportError:
                pass

        path = float(np.sum(np.linalg.norm(np.diff(poses_arr[:, :3, 3], axis=0), axis=1)))
        log.info(
            "Camera trajectory computed: %s | path length %.3f m | failed frames %d",
            poses_arr.shape, path, n_failed,
        )

        return CameraTrajectoryResult(
            poses_world_camera=poses_arr,
            intrinsics=K,
            source="plane-visual-odometry",
            table_normal=ref_normal,
        )

    @staticmethod
    def _frame_homography(
        prev_gray: np.ndarray,
        gray: np.ndarray,
        prev_mask: np.ndarray,
        curr_mask: np.ndarray,
        feature_params: dict,
        lk_params: dict,
    ) -> np.ndarray | None:
        """Homography mapping prev-frame pixels to current-frame pixels, from background only."""
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=prev_mask, **feature_params)
        if prev_pts is None or len(prev_pts) < MIN_TRACKS:
            return None

        next_pts, st_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        back_pts, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, next_pts, None, **lk_params)
        fb_err = np.linalg.norm((prev_pts - back_pts).reshape(-1, 2), axis=1)
        good = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < 1.0)

        # Drop tracks that landed on the hand in the current frame
        nxt = next_pts.reshape(-1, 2)
        h, w = curr_mask.shape
        xi = np.clip(nxt[:, 0].astype(int), 0, w - 1)
        yi = np.clip(nxt[:, 1].astype(int), 0, h - 1)
        good &= curr_mask[yi, xi] > 0

        a = prev_pts.reshape(-1, 2)[good]
        b = nxt[good]
        if len(a) < MIN_TRACKS:
            return None
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.5)
        if H is None or inl is None or int(inl.sum()) < MIN_TRACKS:
            return None
        return H

    @staticmethod
    def _pose_from_homography(
        H_0i: np.ndarray,
        K: np.ndarray,
        d0: float,
        ref_normal: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """World-from-camera pose of frame i from the plane homography H_0->i."""
        n_sol, Rs, ts, ns = cv2.decomposeHomographyMat(H_0i, K)
        if n_sol == 0:
            return None, ref_normal

        # Physical validity: the table must lie in front of the frame-0 camera. With
        # OpenCV's convention n^T X = d (d > 0), a plane facing the camera has n_z > 0.
        candidates = [k for k in range(n_sol) if float(ns[k][2, 0]) > 0]
        if not candidates:
            candidates = list(range(n_sol))

        if ref_normal is None:
            # First solve: prefer the plane most facing the camera (top-down view of a table).
            best = max(candidates, key=lambda k: float(ns[k][2, 0]))
        else:
            # The normal lives in the frame-0 camera frame, so it must stay constant.
            best = max(candidates, key=lambda k: float(ns[k].ravel() @ ref_normal))

        # Only lock the reference normal once motion is large enough to make it well-defined.
        if ref_normal is None and float(np.linalg.norm(ts[best])) > 0.02:
            ref_normal = ns[best].ravel().copy()

        T_i0 = np.eye(4)
        T_i0[:3, :3] = Rs[best]
        T_i0[:3, 3] = ts[best].ravel() * d0      # X_i = R X_0 + t * d0
        return np.linalg.inv(T_i0).astype(np.float32), ref_normal
