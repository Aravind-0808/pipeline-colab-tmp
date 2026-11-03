"""World-Space Fusion and Coordinate Transformation Module.

Hand2Robot stage 4 (world fusion):
Fuses 3D hand reconstructions in camera frame with 6-DoF camera poses in world frame
to produce a temporally consistent, metric 3D hand motion trajectory in world space.

Mathematical Formulation:
For each joint j in [0..20] at frame t:
    p_cam_homogeneous = [x_c, y_c, z_c, 1]^T
    p_world_homogeneous = T_world_camera(t) @ p_cam_homogeneous
    p_world = p_world_homogeneous[:3]

And wrist transformation:
    T_world_wrist(t) = T_world_camera(t) @ T_camera_wrist(t)

Gap handling:
- Gaps of <= MAX_INFILL_GAP missing frames are linearly interpolated (flagged is_infilled).
- Longer gaps are left empty: no hand data is invented.
- Smoothing runs per contiguous segment, never across a long gap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from dfpipeline.retarget.hand_reconstruction import HandReconstruction3D

log = logging.getLogger("hand2robot.world_fusion")

MAX_INFILL_GAP = 10            # frames (~0.33 s at 30 fps)
MAX_WRIST_SPEED_M_S = 3.0      # natural desk manipulation rarely exceeds this


@dataclass
class WorldHandFrame:
    """Single frame fused hand representation in world and camera coordinates."""

    frame_idx: int
    timestamp: float
    hand_side: str
    wrist_world: np.ndarray      # (3,)
    joints_world: np.ndarray     # (21, 3)
    wrist_camera: np.ndarray     # (3,)
    joints_camera: np.ndarray    # (21, 3)
    camera_pose_world: np.ndarray  # (4, 4)
    confidence: float
    is_infilled: bool = False


class WorldSpaceFusion:
    """Fuses camera-space hand geometry with visual-odometry camera poses."""

    def __init__(self, temporal_smoothing: bool = True) -> None:
        self.temporal_smoothing = temporal_smoothing

    def transform_point_cloud(
        self,
        points_camera: np.ndarray,
        T_world_camera: np.ndarray,
    ) -> np.ndarray:
        """Transform an (N, 3) point cloud from camera coordinates to world coordinates."""
        N = points_camera.shape[0]
        ones = np.ones((N, 1), dtype=points_camera.dtype)
        homo_cam = np.hstack([points_camera, ones])  # (N, 4)
        homo_world = homo_cam @ T_world_camera.T  # (N, 4)
        return homo_world[:, :3]

    def fuse_sequence(
        self,
        reconstructions: list[HandReconstruction3D | None],
        camera_poses: np.ndarray,
        fps: float = 24.0,
    ) -> list[WorldHandFrame]:
        """Fuse complete sequence of hand reconstructions with camera poses.

        Args:
            reconstructions: List of HandReconstruction3D (or None if missing) per frame.
            camera_poses: (T, 4, 4) array of world-from-camera poses.
            fps: Frame rate of the video.

        Returns:
            List of WorldHandFrame objects for each valid/infilled frame.
        """
        total_frames = len(reconstructions)
        fused_frames: list[WorldHandFrame] = []

        log.info(
            "Fusing %d hand reconstructions with %d camera poses in world coordinates...",
            total_frames,
            len(camera_poses),
        )

        # 1. Transform each detected frame into world space
        for idx in range(total_frames):
            recon = reconstructions[idx]
            if recon is None:
                continue
            cam_pose = camera_poses[min(idx, len(camera_poses) - 1)]
            joints_world = self.transform_point_cloud(recon.joints_camera, cam_pose)
            fused_frames.append(
                WorldHandFrame(
                    frame_idx=idx,
                    timestamp=idx / fps if fps > 0 else 0.0,
                    hand_side=recon.hand_side,
                    wrist_world=joints_world[0].copy(),
                    joints_world=joints_world,
                    wrist_camera=recon.wrist_camera,
                    joints_camera=recon.joints_camera,
                    camera_pose_world=cam_pose,
                    confidence=recon.confidence,
                    is_infilled=recon.is_infilled,
                )
            )

        # 2. Fill short gaps; leave long gaps empty
        fused_frames = self._infill_short_gaps(fused_frames, camera_poses, fps)

        # 3. Outlier clamp + smoothing, per contiguous segment
        if self.temporal_smoothing:
            smoothed: list[WorldHandFrame] = []
            for segment in self._split_segments(fused_frames):
                smoothed.extend(self._apply_temporal_smoothing(segment))
            fused_frames = smoothed

        n_infilled = sum(f.is_infilled for f in fused_frames)
        log.info(
            "Fused sequence generated: %d world-space frames (%d interpolated, %d missing).",
            len(fused_frames),
            n_infilled,
            total_frames - len(fused_frames),
        )
        return fused_frames

    def _infill_short_gaps(
        self,
        frames: list[WorldHandFrame],
        camera_poses: np.ndarray,
        fps: float,
    ) -> list[WorldHandFrame]:
        """Linearly interpolate world joints across gaps of <= MAX_INFILL_GAP frames."""
        if len(frames) < 2:
            return frames

        out: list[WorldHandFrame] = [frames[0]]
        for a, b in zip(frames, frames[1:]):
            gap = b.frame_idx - a.frame_idx - 1
            if 0 < gap <= MAX_INFILL_GAP:
                for k in range(1, gap + 1):
                    alpha = k / (gap + 1)
                    idx = a.frame_idx + k
                    cam_pose = camera_poses[min(idx, len(camera_poses) - 1)]
                    joints_world = (1 - alpha) * a.joints_world + alpha * b.joints_world
                    # Keep camera-frame data consistent with the interpolated world joints
                    T_cam_world = np.linalg.inv(cam_pose)
                    joints_camera = self.transform_point_cloud(joints_world, T_cam_world)
                    out.append(
                        WorldHandFrame(
                            frame_idx=idx,
                            timestamp=idx / fps if fps > 0 else 0.0,
                            hand_side=a.hand_side,
                            wrist_world=joints_world[0].copy(),
                            joints_world=joints_world.astype(np.float32),
                            wrist_camera=joints_camera[0].astype(np.float32),
                            joints_camera=joints_camera.astype(np.float32),
                            camera_pose_world=cam_pose,
                            confidence=min(a.confidence, b.confidence) * 0.5,
                            is_infilled=True,
                        )
                    )
            out.append(b)
        return out

    @staticmethod
    def _split_segments(frames: list[WorldHandFrame]) -> list[list[WorldHandFrame]]:
        """Split into runs of consecutive frame indices."""
        segments: list[list[WorldHandFrame]] = []
        for f in frames:
            if segments and f.frame_idx == segments[-1][-1].frame_idx + 1:
                segments[-1].append(f)
            else:
                segments.append([f])
        return segments

    def _apply_temporal_smoothing(
        self,
        frames: list[WorldHandFrame],
    ) -> list[WorldHandFrame]:
        """Biomechanical velocity clamp + Savitzky-Golay smoothing on one contiguous segment."""
        T = len(frames)
        if T < 5:
            return frames

        joints_traj = np.array([f.joints_world for f in frames], dtype=np.float32)  # (T, 21, 3)
        timestamps = np.array([f.timestamp for f in frames], dtype=np.float32)

        # 1. Biomechanical outlier clipping. The whole hand is shifted rigidly so
        #    fingers stay attached to the clamped wrist.
        dt = np.clip(np.diff(timestamps), 0.01, 0.1)
        for t in range(1, T):
            step = joints_traj[t, 0] - joints_traj[t - 1, 0]
            dist = float(np.linalg.norm(step))
            max_dist = MAX_WRIST_SPEED_M_S * float(dt[t - 1])
            if dist > max_dist:
                correction = step * (max_dist / dist) - step
                joints_traj[t] += correction

        # 2. Savitzky-Golay polynomial filter (window up to 15, polyorder 2)
        win = min(15, T if T % 2 != 0 else T - 1)
        if win >= 5:
            try:
                from scipy.signal import savgol_filter
                joints_traj = savgol_filter(joints_traj, window_length=win, polyorder=2, axis=0)
            except ImportError:
                log.warning("scipy not installed; skipping Savitzky-Golay smoothing.")

        for idx, f in enumerate(frames):
            f.joints_world = joints_traj[idx].astype(np.float32)
            f.wrist_world = f.joints_world[0].copy()

        return frames
