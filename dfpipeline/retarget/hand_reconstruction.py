"""Lightweight 3D Hand Reconstruction Module.

Hand2Robot stage 2 (3D hand):
Reconstructs metric 3D hand pose and 21 MANO-ordered joints in camera space from detected hands
across video frames with:
1. PnP pose fit of MediaPipe metric world landmarks to the 2D keypoints
   (fallback: palm-length depth scaling)
2. One-Euro (1€) Adaptive Filtering for zero-jitter and zero-lag tracking

Topology (21 MANO Joints):
  0: Wrist
  1-4: Thumb (CMC, MCP, IP, Tip)
  5-8: Index (MCP, PIP, DIP, Tip)
  9-12: Middle (MCP, PIP, DIP, Tip)
  13-16: Ring (MCP, PIP, DIP, Tip)
  17-20: Pinky (MCP, PIP, DIP, Tip)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from dfpipeline.retarget.hand_detection import HandDetection

log = logging.getLogger("hand2robot.hand_reconstruction")

MANO_JOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

# Anatomical bone lengths (in meters) for standard adult human hand
# Palm: Wrist to Middle MCP = ~0.095m
CANONICAL_PALM_LENGTH_M = 0.095

# Plausible hand distance from an egocentric / handheld camera
MIN_HAND_DEPTH_M = 0.10
MAX_HAND_DEPTH_M = 2.00


class OneEuroFilter:
    """One-Euro adaptive low-pass filter for eliminating jitter while preserving zero lag."""

    def __init__(
        self,
        t0: float,
        x0: np.ndarray,
        min_cutoff: float = 1.2,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0.copy()
        self.dx_prev = np.zeros_like(x0)
        self.t_prev = t0

    def filter(self, t: float, x: np.ndarray) -> np.ndarray:
        dt = t - self.t_prev
        if dt <= 1e-5:
            return self.x_prev

        # 1. Filter derivative
        a_d = self._alpha(dt, self.d_cutoff)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # 2. Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    @staticmethod
    def _alpha(dt: float, cutoff: np.ndarray | float) -> np.ndarray | float:
        r = 2.0 * np.pi * cutoff * dt
        return r / (r + 1.0)


@dataclass
class HandReconstruction3D:
    """Reconstructed 3D hand output for a single frame."""

    hand_side: str
    joints_camera: np.ndarray    # Shape: (21, 3) in meters
    wrist_camera: np.ndarray     # Shape: (3,) in meters
    pose_orientation: np.ndarray # Shape: (3, 3) rotation matrix
    confidence: float
    is_infilled: bool = False
    keypoints_2d: np.ndarray | None = None  # (21, 2) in pixels


class HandReconstructor:
    """Geometric 3D hand reconstruction: PnP on MediaPipe metric landmarks + One-Euro filtering."""

    def __init__(self) -> None:
        self.one_euro_filter: OneEuroFilter | None = None
        self._prev_rvec: np.ndarray | None = None
        self._prev_tvec: np.ndarray | None = None

    def reconstruct_from_detection(
        self,
        frame_idx: int,
        timestamp: float,
        detection: HandDetection,
        camera_intrinsics: np.ndarray,
        reset_filter: bool = False,
    ) -> HandReconstruction3D | None:
        """Reconstruct 21 3D joints in metric camera coordinates.

        Returns None when no geometrically valid reconstruction can be made,
        so the frame is treated as missing instead of producing a fake hand.
        """
        if detection.keypoints_2d is None or len(detection.keypoints_2d) != 21:
            return None

        # After a tracking gap, restart the filter so it doesn't drag the hand
        # from its old position (causes a visible "slide" after re-detection).
        if reset_filter:
            self.one_euro_filter = None
            self._prev_rvec = None
            self._prev_tvec = None

        joints_camera = self._solve_pnp(detection, camera_intrinsics)
        if joints_camera is None:
            joints_camera = self._palm_scale_unproject(detection, camera_intrinsics)
        if joints_camera is None:
            return None

        # Apply One-Euro Adaptive Filter for smooth, high-accuracy tracking
        if self.one_euro_filter is None:
            self.one_euro_filter = OneEuroFilter(
                t0=timestamp,
                x0=joints_camera,
                min_cutoff=3.0,  # Hz: jitter suppression at rest
                beta=20.0,  # joints are in meters (m/s speeds), not pixels: fast motion must not lag
            )
            filtered_joints = joints_camera
        else:
            filtered_joints = self.one_euro_filter.filter(timestamp, joints_camera)

        wrist_pos = filtered_joints[0].copy()

        # Compute hand orientation from wrist, index MCP (5), and pinky MCP (17)
        v_index = filtered_joints[5] - filtered_joints[0]
        v_pinky = filtered_joints[17] - filtered_joints[0]
        v_normal = np.cross(v_index, v_pinky)
        norm_val = np.linalg.norm(v_normal)
        if norm_val > 1e-6:
            v_normal /= norm_val
        else:
            v_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        rot_matrix = np.eye(3, dtype=np.float32)
        rot_matrix[:, 2] = v_normal

        return HandReconstruction3D(
            hand_side=detection.hand_side,
            joints_camera=filtered_joints,
            wrist_camera=wrist_pos,
            pose_orientation=rot_matrix,
            confidence=detection.confidence,
            is_infilled=False,
            keypoints_2d=detection.keypoints_2d,
        )

    def _solve_pnp(
        self,
        detection: HandDetection,
        K: np.ndarray,
    ) -> np.ndarray | None:
        """Metric hand pose from MediaPipe world landmarks (meters) + 2D keypoints.

        All 21 correspondences constrain depth, so a tilted palm no longer
        makes the hand jump toward/away from the camera.
        """
        obj = detection.landmarks_world
        if obj is None or len(obj) != 21:
            return None
        obj = obj.astype(np.float64)
        img = detection.keypoints_2d.astype(np.float64)
        K64 = K.astype(np.float64)
        dist = np.zeros(4)

        try:
            if self._prev_rvec is not None:
                ok, rvec, tvec = cv2.solvePnP(
                    obj, img, K64, dist,
                    rvec=self._prev_rvec.copy(), tvec=self._prev_tvec.copy(),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                ok, rvec, tvec = cv2.solvePnP(obj, img, K64, dist, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return None
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K64, dist, rvec, tvec)
        except cv2.error:
            return None

        # Reject physically implausible or poorly fitting solutions.
        z = float(tvec[2, 0])
        if not (MIN_HAND_DEPTH_M <= z <= MAX_HAND_DEPTH_M):
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K64, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
        hand_px = float(np.linalg.norm(img.max(axis=0) - img.min(axis=0)))
        if err > max(8.0, 0.15 * hand_px):
            return None

        self._prev_rvec, self._prev_tvec = rvec, tvec
        R, _ = cv2.Rodrigues(rvec)
        z = ((R @ obj.T).T + tvec.reshape(1, 3))[:, 2]
        # Keep the metric per-joint depth from PnP, but place each joint exactly on
        # its detected 2D keypoint so the skeleton aligns with the real fingers.
        fx, fy, cx, cy = K64[0, 0], K64[1, 1], K64[0, 2], K64[1, 2]
        joints = np.stack([(img[:, 0] - cx) * z / fx, (img[:, 1] - cy) * z / fy, z], axis=1)
        return joints.astype(np.float32)

    def _palm_scale_unproject(
        self,
        detection: HandDetection,
        K: np.ndarray,
    ) -> np.ndarray | None:
        """Fallback: depth from the rigid wristâ†’middle-MCP palm length."""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        kpts_2d = detection.keypoints_2d
        palm_pixel_dist = float(np.linalg.norm(kpts_2d[0] - kpts_2d[9]))
        if palm_pixel_dist < 15.0:
            return None
        base_zc = float(np.clip((fy * CANONICAL_PALM_LENGTH_M) / palm_pixel_dist, 0.30, 1.35))

        joints_camera = np.zeros((21, 3), dtype=np.float32)
        lms_3d_norm = detection.landmarks_3d_norm
        for j in range(21):
            u, v = kpts_2d[j]
            dz = float(lms_3d_norm[j, 2]) * 0.22 if lms_3d_norm is not None else 0.0
            zj = base_zc + dz
            joints_camera[j] = [(u - cx) * zj / fx, (v - cy) * zj / fy, zj]
        return joints_camera
