"""Data Export Module for Machine-Readable Trajectories.

Hand2Robot trajectory export:
Exports structured trajectories to:
1. output/trajectory.json
2. output/trajectory.npz

Conforms strictly to the required schema:
{
  "frame": 120,
  "timestamp": 4.0,
  "hand": "right",
  "wrist_world": [x, y, z],
  "joints_world": [
    [x, y, z],
    ... (21 joints)
  ],
  "confidence": 0.95
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from dfpipeline.retarget.hand_reconstruction import MANO_JOINT_NAMES
from dfpipeline.retarget.world_fusion import WorldHandFrame

log = logging.getLogger("hand2robot.export")


class TrajectoryExporter:
    """Exports world-space hand trajectories to JSON and NPZ formats."""

    def __init__(self, output_dir: str | Path = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        frames: list[WorldHandFrame],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """Export trajectory data to trajectory.json and trajectory.npz.

        Args:
            frames: Sequence of WorldHandFrame objects.
            metadata: Optional additional pipeline metadata.

        Returns:
            Tuple of (json_path, npz_path).
        """
        if not frames:
            raise ValueError("No frames provided for trajectory export.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "trajectory.json"
        npz_path = self.output_dir / "trajectory.npz"

        # 1. Prepare JSON structure
        json_records = []
        for f in frames:
            record = {
                "frame": int(f.frame_idx),
                "timestamp": float(round(f.timestamp, 4)),
                "hand": str(f.hand_side),
                "wrist_world": [float(round(val, 5)) for val in f.wrist_world.tolist()],
                "joints_world": [
                    [float(round(coord, 5)) for coord in joint]
                    for joint in f.joints_world.tolist()
                ],
                "confidence": float(round(f.confidence, 4)),
                "infilled": bool(f.is_infilled),
            }
            json_records.append(record)

        payload = {
            "version": "hand2robot-1.0",
            "pipeline": "MediaPipe + Geometric 3D Hand + Visual Odometry World-Space Fusion",
            "joint_order_documentation": {
                idx: name for idx, name in enumerate(MANO_JOINT_NAMES)
            },
            "total_frames_reconstructed": len(json_records),
            "metadata": metadata or {},
            "trajectory": json_records,
        }

        with open(json_path, "w", encoding="utf-8") as f_out:
            json.dump(payload, f_out, indent=2)

        log.info("Exported JSON trajectory to %s (%d records)", json_path, len(json_records))

        # 2. Prepare NPZ structure
        frame_indices = np.array([f.frame_idx for f in frames], dtype=np.int32)
        timestamps = np.array([f.timestamp for f in frames], dtype=np.float32)
        wrist_world = np.stack([f.wrist_world for f in frames], axis=0).astype(np.float32)
        joints_world = np.stack([f.joints_world for f in frames], axis=0).astype(np.float32)
        wrist_camera = np.stack([f.wrist_camera for f in frames], axis=0).astype(np.float32)
        joints_camera = np.stack([f.joints_camera for f in frames], axis=0).astype(np.float32)
        camera_poses = np.stack([f.camera_pose_world for f in frames], axis=0).astype(np.float32)
        confidences = np.array([f.confidence for f in frames], dtype=np.float32)
        infilled = np.array([f.is_infilled for f in frames], dtype=bool)

        np.savez_compressed(
            npz_path,
            frames=frame_indices,
            timestamps=timestamps,
            wrist_world=wrist_world,
            joints_world=joints_world,
            wrist_camera=wrist_camera,
            joints_camera=joints_camera,
            camera_poses=camera_poses,
            confidence=confidences,
            infilled=infilled,
            joint_names=np.array(MANO_JOINT_NAMES),
        )

        log.info("Exported NPZ trajectory to %s", npz_path)
        return json_path, npz_path
