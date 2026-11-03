"""Hand2Robot pipeline: the orchestrator, vendored from the Hand2Robot POC.

One clip of a human doing a table-top task in, one SO-101 joint trajectory out.
Called from the `retarget.clip` job; see dfpipeline/retarget/service.py.

Pipeline Stages:
1. Video Verification & Extraction
2. MediaPipe Hand Detection & Tracking
3. Geometric 3D Hand Reconstruction (21 MANO-ordered Joints)
4. Visual-Odometry Camera/World Reconstruction (SE(3) Poses)
5. World-Space Coordinate Alignment & Fusion
6. Trajectory Export (JSON & NPZ)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dfpipeline.retarget import lens as lenses
from dfpipeline.retarget.camera_reconstruction import CameraPoseEstimator
from dfpipeline.retarget.export import TrajectoryExporter
from dfpipeline.retarget.hand_detection import SingleHandTracker, TwoHandTracker
from dfpipeline.retarget.hand_reconstruction import HandReconstructor
from dfpipeline.retarget.hand_wilor import make_detector
from dfpipeline.retarget.retarget_so101 import (
    ARM_SIDES,
    SO101Retargeter,
    export_pair,
    retarget_pair,
)
from dfpipeline.retarget.world_fusion import MAX_INFILL_GAP, WorldSpaceFusion

log = logging.getLogger("hand2robot.pipeline")


def _export_object(out_dir: Path, track, info: dict, fps: float) -> None:
    """object_trajectory.json: the manipulated object per video frame (world = frame-0 camera)."""
    import json

    base = info["base_world"]
    records = [
        {
            "frame": k,
            "timestamp": round(k / fps, 4),
            "bbox_px": [round(float(v), 1) for v in track.bbox[k]],
            "detected": bool(track.detected[k]),
            "state": "held" if track.held[k] else "resting",
            "base_world_m": [round(float(v), 4) for v in base[k]],
        }
        for k in range(len(base))
    ]
    payload = {
        "label": info["label"],
        "label_note": "detector class (COCO); a cup seen from above is often reported as bowl/frisbee",
        "diameter_m": info["diameter_m"],
        "height_m": info["height_m_prior"],
        "height_note": "prior for the label: a top-down view shows only the top",
        "mass_kg_estimate": info["mass_kg_estimate"],
        "mass_note": "estimate from the label; mass cannot be measured from video",
        "picks_s": [round(k / fps, 2) for k in info["picks"]],
        "places_s": [round(k / fps, 2) for k in info["places"]],
        "frames": records,
    }
    with open(out_dir / "object_trajectory.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh)



def probe_stream(path: Path) -> tuple[int, float] | None:
    """The clip's real frame count and duration, by demuxing it.

    OpenCV reads both from the container's declared rate, and a WebM written by
    a browser's MediaRecorder declares whatever it likes. Every clip in the
    first real batch said 60 fps and 498 frames while holding 250 frames of
    8.3 seconds — exactly half. Everything downstream inherited that: the
    overlay played at double speed, hand coverage was halved against a
    denominator twice the truth, and per-frame timestamps compressed the hand
    path into half its real duration, which doubled the retiming the arm was
    judged on.

    So the frames are counted rather than asked for. It is a demux, not a
    decode — 40ms on these clips — and it is the only number that cannot lie.

    Counted with PyAV, packet by packet. ffmpeg's own log was read for this
    before, and ffmpeg 7 stopped printing a frame count for a stream copy: the
    probe failed quietly and a 924-frame WebM was taken at its container's
    word, 31,366 frames.
    """
    counted = _count_packets(path)
    if counted is not None:
        return counted
    import re
    import subprocess

    from dfpipeline.video import ffmpeg_exe

    try:
        out = subprocess.run(
            [ffmpeg_exe(), "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=120,
        ).stderr
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not probe %s (%s); falling back to what the container claims", path.name, exc)
        return None

    frames = None
    for match in re.finditer(r"frame=\s*(\d+)", out):
        frames = int(match.group(1))
    stamp = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    if not frames or not stamp:
        return None
    seconds = int(stamp.group(1)) * 3600 + int(stamp.group(2)) * 60 + float(stamp.group(3))
    return (frames, seconds) if seconds > 0 else None


def _count_packets(path: Path) -> tuple[int, float] | None:
    """Video frames and duration from the packets themselves; None if unreadable."""
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            pts = sorted(p.pts for p in container.demux(stream) if p.size and p.pts is not None)
            base = stream.time_base
    except Exception as exc:  # noqa: BLE001 - any unreadable file falls back
        log.warning("could not count the frames of %s with PyAV (%s)", path.name, exc)
        return None
    if len(pts) < 2 or base is None:
        return None
    n = len(pts)
    # The last frame lasts one frame too: span of n-1 gaps, scaled to n.
    seconds = float((pts[-1] - pts[0]) * base) * n / (n - 1)
    return (n, seconds) if seconds > 0 else None


def run_hand_stage(
    input_video: str | Path,
    output_dir: str | Path,
    progress_callback: Any = None,
    static_camera: bool = False,
    hands: int = 1,
    camera: dict | None = None,
) -> int:
    """Reconstruct the human hand, and the object it manipulates, in 3D.

    `camera`: what the phone said about its camera (lens.py), when it did —
    the lens, and which way is up.

    `hands=2` for a task done with both hands: each is followed on its own
    track and written to its own folder, `left/` and `right/` (see
    `_run_two_hands`).

    Everything here is about the person and nothing about any robot: 21 hand
    joints in world coordinates, the camera's own motion, and the object with
    its size and the moments it is held. The same answer serves every arm the
    clip is ever exported for, so it is done once per clip and kept.

    It is also the expensive half — MediaPipe over every frame, twice — which
    is why it runs when a clip is approved rather than when someone is waiting
    on an export.
    """
    start_time = time.perf_counter()
    input_path = Path(input_video)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validation: Verify video file
    if not input_path.exists():
        log.error("Input video file does not exist: %s", input_path)
        return 1

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        log.error("Unsupported video format or unreadable file: %s", input_path)
        return 1

    # Counted, not asked for — see probe_stream. What the container declares is
    # only a fallback now, for a file ffmpeg cannot demux.
    probed = probe_stream(input_path)
    if probed is not None:
        total_frames, duration_s = probed
        fps = total_frames / duration_s
        declared = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if declared and abs(declared - fps) / fps > 0.05:
            log.info(
                "%s declares %.1f fps but holds %d frames in %.2fs — using %.2f",
                input_path.name, declared, total_frames, duration_s, fps,
            )
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        raw_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        # Sanitize anomalous FPS (e.g. 1000 FPS timebase in browser-recorded WebM)
        if raw_fps < 5.0 or raw_fps > 65.0:
            log.warning(
                "Anomalous video FPS detected: %.2f (common in browser WebM MediaRecorder). Normalizing to 30.0 FPS.",
                raw_fps,
            )
            fps = 30.0
        else:
            fps = float(raw_fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
    duration = total_frames / fps if fps > 0 else 0.0

    log.info(
        "Input Video: %s | Frames: %d | FPS: %.1f | Res: %dx%d | Duration: %.2fs",
        input_path.name,
        total_frames,
        fps,
        width,
        height,
        duration,
    )

    if total_frames <= 0:
        log.error("Video contains zero valid frames: %s", input_path)
        cap.release()
        return 1

    # The lens: from the phone where it said, else the depth model's estimate
    # (lens.py, scene.py). Never guessed: a guessed lens put the hand 0.9 m
    # away where it was 0.65, and every distance after it was wrong.
    lens = lenses.resolve(camera, width, height)
    if lens.source == "guess":
        from dfpipeline.retarget import scene

        estimated = scene.estimate_hfov(input_path, total_frames)
        measured = lenses.from_hfov(estimated, width, height, "depth-model") if estimated else None
        if measured is None:
            log.error("The lens could not be measured: the phone sent no camera data and "
                      "the depth model gave no field of view.")
            cap.release()
            return 1
        lens = measured
    K = lens.K
    up = lenses.up_in_image(camera)
    log.info("Lens: %s, %.0f° across%s; up %s", lens.source, lens.hfov_deg,
             " (stabilised)" if lens.stabilized else "", "known" if up is not None else "unknown")

    if hands == 2:
        cap.release()
        # Both hands do the task, so neither holds the phone: it is on a stand.
        # The camera's own motion is estimated for a phone held in the free hand,
        # and on a still camera that estimate is only error — on a plain cloth,
        # 30 cm of motion that never happened, which put a whole arm 22 cm off
        # its hand's path.
        return _run_two_hands(input_path, out_dir, fps, total_frames, (width, height), lens,
                              progress_callback, True, start_time, up)

    log.info("=" * 65)
    log.info("HAND2ROBOT: video -> 3D hand & object -> SO-101")
    log.info("Pipeline: WiLoR + depth model (GPU) -> 3D Hand -> Visual Odometry -> World Fusion")
    log.info("=" * 65)

    # 2. Initialize Pipeline Modules
    hand_detector = make_detector()
    hand_reconstructor = HandReconstructor()
    camera_estimator = CameraPoseEstimator()
    world_fusion = WorldSpaceFusion(temporal_smoothing=True)
    exporter = TrajectoryExporter(output_dir=out_dir)

    # 3. Stage 1 & 2: Process frames for Hand Detection & 3D Reconstruction
    log.info("Stage 1 & 2: Running WiLoR Hand Detection & 3D Reconstruction...")
    reconstructions: list = []
    hand_bboxes: dict[int, list[list[float]]] = {}
    pinch_px: dict[int, np.ndarray] = {}
    tracker = SingleHandTracker(frame_diag_px=float(np.hypot(width, height)))
    hand_kp: dict[int, np.ndarray] = {}  # the tracked hand's 2D points, for its holds (grasp.py)
    seen: dict[int, list] = {}  # frame -> [(2D points, reconstruction)], for the depth model (scene.py)
    hands_detected_count = 0
    frames_since_valid = 0

    cap.release()
    all_detections = hand_detector.detect_video(input_path, fps, progress_callback)

    for frame_idx, detections in enumerate(all_detections):
        timestamp = frame_idx / fps
        if detections:
            # Every visible hand is masked out of camera odometry
            hand_bboxes[frame_idx] = [d.bbox for d in detections]

        recon = None
        best_det = tracker.select(detections)
        if best_det is not None:
            hand_kp[frame_idx] = best_det.keypoints_2d
            pinch_px[frame_idx] = 0.5 * (best_det.keypoints_2d[4] + best_det.keypoints_2d[8])
            recon = hand_reconstructor.reconstruct_from_detection(
                frame_idx=frame_idx,
                timestamp=timestamp,
                detection=best_det,
                camera_intrinsics=K,
                reset_filter=frames_since_valid > MAX_INFILL_GAP,
            )

        if recon is not None:
            hands_detected_count += 1
            frames_since_valid = 0
            seen[frame_idx] = [(best_det.keypoints_2d, recon)]
        else:
            frames_since_valid += 1
        reconstructions.append(recon)

    log.info(
        "Hand Tracking complete: %d / %d frames reconstructed (%.1f%%), tracked hand: %s",
        hands_detected_count,
        total_frames,
        (hands_detected_count / max(total_frames, 1)) * 100,
        tracker.side,
    )

    if hands_detected_count == 0:
        log.error("Tracking Failure: No human hand detected in the video.")
        return 1

    # Real distances, where a depth model is set (scene.py): the hand's true
    # size and the table. Without one, both are estimated as before.
    if progress_callback:
        progress_callback(52.0, "Stage 2/3: Measuring depth...")
    scene_found = _measure_scene(input_path, lens, up, total_frames, seen, hand_bboxes)

    # 4. Stage 3: Camera & World Reconstruction (Visual Odometry)
    log.info("Stage 3: Running Visual-Odometry Camera/World Pose Reconstruction...")
    if progress_callback:
        progress_callback(55.0, "Stage 2/3: Estimating 6-DoF Camera Trajectory (Visual Odometry)...")
    plane_distance = _plane_distance(scene_found, reconstructions)
    camera_result = camera_estimator.estimate_trajectory(
        input_path,
        hand_bboxes=hand_bboxes,
        static_camera=static_camera,
        plane_distance_m=plane_distance,
        K=K,
    )
    if scene_found is not None and scene_found.table is not None:
        # Measured beats inferred from how the table slid between frames.
        camera_result.table_normal = scene_found.table.normal.astype(np.float32)
    camera_poses = camera_result.poses_world_camera

    if len(camera_poses) == 0:
        log.error("Camera reconstruction failure: No camera poses computed.")
        return 1

    # 5. Stage 4: World-Space Fusion & Coordinate Alignment
    log.info("Stage 4: Aligning Coordinates and Fusing into World Space...")
    if progress_callback:
        progress_callback(68.0, "Stage 2/3: Aligning & Fusing into World Space...")
    fused_frames = world_fusion.fuse_sequence(
        reconstructions=reconstructions,
        camera_poses=camera_poses,
        fps=fps,
    )

    if not fused_frames:
        log.error("World fusion produced zero valid frames.")
        return 1

    # Verify 3D coordinates validity
    sample_wrist = fused_frames[0].wrist_world
    if np.any(np.isnan(sample_wrist)) or np.any(np.isinf(sample_wrist)):
        log.error("Invalid 3D coordinates detected (NaN or Inf values).")
        return 1

    # 5b. Manipulated-object tracking: what the object is, where it is, its size, and when it is
    # held (pick / place events), in the same world frame as the hand.
    object_track, object_info, objects = None, None, None
    try:
        from dfpipeline.retarget.object_tracking import ObjectTracker, lift_to_world

        objects = ObjectTracker()
        object_track = objects.track(input_path, hand_bboxes, pinch_px, fps, progress_callback)
        if object_track is not None:
            pinch_world = {f.frame_idx: 0.5 * (f.joints_world[4] + f.joints_world[8]) for f in fused_frames}
            object_info = lift_to_world(
                object_track, camera_poses, K, camera_result.table_normal, plane_distance or 0.5, pinch_world
            )
            _export_object(out_dir, object_track, object_info, fps)
    except Exception as exc:
        log.warning("Object tracking skipped: %s", exc)

    # 6. Stage 5: Export Trajectories (JSON & NPZ)
    log.info("Stage 5: Exporting machine-readable trajectories...")
    if progress_callback:
        progress_callback(74.0, "Stage 2/3: Serializing Trajectory Dataset (JSON & NPZ)...")
    metadata = {
        "video_file": input_path.name,
        "fps": fps,
        "total_frames": total_frames,
        "reconstructed_frames": len(fused_frames),
        # From the first frame the hand was reconstructed in to the last: the
        # reach and the task. Frames before the hand comes in and after it
        # leaves are not missing hand, and coverage is measured over this span.
        "hand_first_frame": min((f.frame_idx for f in fused_frames), default=None),
        "hand_last_frame": max((f.frame_idx for f in fused_frames), default=None),
        "camera_reconstruction_source": camera_result.source,
        "hand_model": "MANO-21",
        "coordinate_systems": {
            "camera_frame": "+X: Right, +Y: Down, +Z: Forward (Optical Axis)",
            "world_frame": "+X: Right, +Y: Down, +Z: Forward, relative to initial camera pose at t=0",
        },
    }
    json_path, npz_path = exporter.export(fused_frames, metadata=metadata)
    _write_holds(out_dir, hand_kp, objects.candidates if objects else [], fused_frames, total_frames, fps, K)

    # What the robot stage needs and the human-readable exports do not carry:
    # the table plane, the distance to it, and the object as the retargeter
    # wants it rather than as a reader wants it. Written here so the arm can be
    # solved later, on its own, without looking at the video again.
    _write_hand_stage(out_dir, fps, total_frames, (width, height), camera_result, plane_distance, object_info,
                      perception=_perception(lens, up, scene_found))

    log.info(
        "Hand stage complete in %.1fs: %d of %d frames, object in %s",
        time.perf_counter() - start_time,
        len(fused_frames),
        total_frames,
        f"{object_info and 'yes' or 'no'}",
    )
    return 0


def run_robot_stage(
    hand_dir: str | Path,
    output_dir: str | Path | None = None,
) -> int:
    """Solve the arm for a hand trajectory that has already been reconstructed.

    The hand stage is the same whatever robot the clip is later exported for —
    a person's hand in 3D is a person's hand — and it is the expensive half.
    This is the half that is per-robot, so it reads what that stage left behind
    instead of watching the video again.
    """
    import json as _json

    hand_dir = Path(hand_dir)
    out_dir = Path(output_dir) if output_dir else hand_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = _json.loads((hand_dir / "hand_stage.json").read_text(encoding="utf-8"))
    if bundle.get("hands"):
        return _robot_stage_two_arms(hand_dir, out_dir, bundle)
    frames = _load_hand_frames(hand_dir)
    if not frames:
        log.error("The hand trajectory has no frames to retarget.")
        return 1

    table_normal = np.asarray(bundle["table_normal"], dtype=np.float64) if bundle.get("table_normal") else None
    object_info = _load_object_info(bundle)

    log.info("Stage 5b: Retargeting %d hand frames to SO-101...", len(frames))
    try:
        retargeter = SO101Retargeter()
        from dfpipeline.retarget import grasp

        robot_frames = retargeter.retarget(
            frames,
            table_normal_cam=table_normal,
            table_distance_m=bundle.get("table_distance_m") or 0.5,
            object_info=object_info,
            grip=grasp.read_grip(hand_dir),
        )
        retargeter.export(robot_frames, out_dir)
    except Exception as exc:
        log.error("SO-101 retargeting failed: %s", exc)
        return 1

    return 0


def _write_holds(folder, keypoints, candidates, frames, total_frames, fps, K, holds=None) -> None:
    """When this hand held something, and how wide it was: the gripper's
    open and close (grasp.py), kept beside its trajectory. Best effort: the
    robot stage falls back to the fingertips without it."""
    from dfpipeline.retarget import grasp

    try:
        holds = holds or grasp.find_holds(keypoints, candidates, total_frames, fps)
        depth = {f.frame_idx: float(0.5 * (f.joints_camera[4, 2] + f.joints_camera[8, 2])) for f in frames}
        width = grasp.object_width_m(holds, depth, float(K[0, 0]))
        grasp.write_grip(folder, holds, width)
        log.info("Holds in %s: %s, object %s m", Path(folder).name,
                 [(a, b) for a, b in grasp._runs(holds.held)], width and round(width, 3))
    except Exception as exc:  # noqa: BLE001 - the gripper falls back to the fingertips
        log.warning("Holds not found: %s", exc)


def _measure_scene(video: Path, lens, up, total_frames: int, seen: dict[int, list],
                   boxes: dict[int, list]):
    """The depth model's measurements (scene.py), with the hand's scale
    already applied to every reconstruction. A depth model that fails fails
    the clip: there is no measuring without it.

    The scale multiplies each joint's camera coordinates: the hand slides
    along its own lines of sight and stays exactly on its pixels.
    """
    from dfpipeline.retarget import scene

    hands = {i: [(kp, r.joints_camera) for kp, r in pairs] for i, pairs in seen.items()}
    found = scene.measure(video, lens.K, lens.hfov_deg, total_frames, hands, boxes, up)
    if found.hand_scale is not None:
        scaled = {id(r): r for pairs in seen.values() for _, r in pairs}
        for r in scaled.values():
            r.joints_camera = (r.joints_camera * found.hand_scale).astype(np.float32)
            r.wrist_camera = r.joints_camera[0].copy()
    return found


def _plane_distance(found, reconstructions) -> float | None:
    """The camera-to-table distance in frame 0: measured, else the hand's
    working depth (the farthest joints, where the hand meets the table)."""
    if found is not None and found.table is not None:
        return float(found.table.distance)
    joint_depths = [float(r.joints_camera[:, 2].max()) for r in reconstructions if r is not None]
    return float(np.percentile(joint_depths, 75)) if joint_depths else None


def _perception(lens, up, found) -> dict:
    """How this clip's distances were made, for the bundle and the review
    screens: the lens from the phone or the depth model, the table and the
    hand's scale measured or not."""
    return {
        "hand_model": "wilor",
        "lens_source": lens.source,
        "hfov_deg": round(lens.hfov_deg, 1),
        "stabilized": lens.stabilized,
        "up_known": up is not None,
        **found.summary(),
    }


def _write_hand_stage(out_dir, fps, total_frames, size, camera_result, plane_distance, object_info,
                      hands: list[str] | None = None, perception: dict | None = None) -> None:
    """The handover file: everything the arm needs that the readable exports drop.

    `hands`, for a two-hand clip: the arms, each hand's trajectory in the
    folder of that name. Absent for one hand, whose files are at the top.
    `perception`: how the distances were made (_perception).
    """
    import json as _json

    normal = getattr(camera_result, "table_normal", None)
    payload = {
        "version": "hand2robot-1.0",
        "fps": float(fps),
        "total_frames": int(total_frames),
        "width": int(size[0]),
        "height": int(size[1]),
        "table_normal": [float(v) for v in np.asarray(normal).ravel()] if normal is not None else None,
        "table_distance_m": float(plane_distance) if plane_distance else None,
        "camera_source": getattr(camera_result, "source", None),
    }
    if perception:
        payload["perception"] = perception
    if hands:
        payload["hands"] = list(hands)
    if object_info:
        payload["object"] = {
            "label": object_info["label"],
            "diameter_m": float(object_info["diameter_m"]),
            "height_m_prior": float(object_info["height_m_prior"]),
            "mass_kg_estimate": float(object_info["mass_kg_estimate"]),
            "base_world": np.asarray(object_info["base_world"], dtype=float).tolist(),
            "held": [bool(v) for v in np.asarray(object_info["held"]).ravel()],
            "picks": [int(v) for v in object_info["picks"]],
            "places": [int(v) for v in object_info["places"]],
        }
    (out_dir / "hand_stage.json").write_text(_json.dumps(payload), encoding="utf-8")


class _HandFrame:
    """The three fields the robot stage reads off a fused frame.

    Deliberately not the whole WorldHandFrame: the camera-frame copies and the
    poses are for drawing the overlay, and nothing downstream of the hand asks
    for them. If the retargeter ever does, it will fail loudly here rather than
    quietly work on a default.
    """

    __slots__ = ("frame_idx", "timestamp", "joints_world")

    def __init__(self, frame_idx: int, timestamp: float, joints_world) -> None:
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.joints_world = joints_world


def _load_hand_frames(hand_dir: Path) -> list:
    """Rebuild the hand path from trajectory.npz, or the JSON if it is absent."""
    npz = hand_dir / "trajectory.npz"
    if npz.is_file():
        data = np.load(npz)
        return [
            _HandFrame(int(i), float(ts), np.asarray(j, dtype=np.float64))
            for i, ts, j in zip(data["frames"], data["timestamps"], data["joints_world"])
        ]

    import json as _json

    records = _json.loads((hand_dir / "trajectory.json").read_text(encoding="utf-8"))["trajectory"]
    return [
        _HandFrame(int(r["frame"]), float(r["timestamp"]), np.asarray(r["joints_world"], dtype=np.float64))
        for r in records
    ]


def _load_object_info(bundle: dict) -> dict | None:
    obj = bundle.get("object")
    if not obj:
        return None
    return {
        "label": obj["label"],
        "diameter_m": obj["diameter_m"],
        "height_m_prior": obj["height_m_prior"],
        "mass_kg_estimate": obj["mass_kg_estimate"],
        "base_world": np.asarray(obj["base_world"], dtype=float),
        "held": np.asarray(obj["held"], dtype=bool),
        "picks": list(obj["picks"]),
        "places": list(obj["places"]),
    }


def _run_two_hands(input_path, out_dir, fps, total_frames, size, lens, progress_callback,
                   static_camera, start_time, up=None) -> int:
    """The hand stage for a two-hand clip.

    The same stages as for one hand, done for each hand on its own track
    (`TwoHandTracker`): its own 3D reconstruction and smoothing, so neither
    hand's filter ever runs across into the other. The camera and the object
    are one for both. Each hand's trajectory is written to `left/` or `right/`
    — named by where it worked in the picture — with the same files and
    metadata a lone hand has at the top. A hand never seen gets no folder; the
    gate says so.
    """
    width, height = size
    K = lens.K
    log.info("HAND2ROBOT, both hands: video -> two 3D hands & object -> two SO-101 arms")
    detections = make_detector().detect_video(input_path, fps, progress_callback, hands=2)

    tracker = TwoHandTracker(frame_diag_px=float(np.hypot(width, height)))
    rebuilders = [HandReconstructor(), HandReconstructor()]
    hand_kp: list[dict[int, np.ndarray]] = [{}, {}]  # each hand's 2D points, for its holds
    recons: list[list] = [[], []]
    since_valid = [0, 0]
    hand_bboxes: dict[int, list[list[float]]] = {}
    pinch_px: dict[int, np.ndarray] = {}
    seen: dict[int, list] = {}  # frame -> [(2D points, reconstruction)] of both hands (scene.py)
    for frame_idx, found in enumerate(detections):
        if found:
            hand_bboxes[frame_idx] = [d.bbox for d in found]
        pinches = []
        for k, det in enumerate(tracker.select(found)):
            recon = None
            if det is not None:
                hand_kp[k][frame_idx] = det.keypoints_2d
                pinches.append(0.5 * (det.keypoints_2d[4] + det.keypoints_2d[8]))
                recon = rebuilders[k].reconstruct_from_detection(
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    detection=det,
                    camera_intrinsics=K,
                    reset_filter=since_valid[k] > MAX_INFILL_GAP,
                )
            since_valid[k] = 0 if recon is not None else since_valid[k] + 1
            if recon is not None:
                seen.setdefault(frame_idx, []).append((det.keypoints_2d, recon))
            recons[k].append(recon)
        if pinches:
            pinch_px[frame_idx] = np.stack(pinches)

    sides = tracker.sides()
    counts = {side: sum(r is not None for r in recons[k]) for k, side in enumerate(sides)}
    log.info("Both hands: %s frames reconstructed of %d", counts, total_frames)
    if not any(counts.values()):
        log.error("Tracking Failure: No human hand detected in the video.")
        return 1

    if progress_callback:
        progress_callback(52.0, "Stage 2/3: Measuring depth...")
    scene_found = _measure_scene(input_path, lens, up, total_frames, seen, hand_bboxes)

    if progress_callback:
        progress_callback(55.0, "Stage 2/3: Estimating 6-DoF Camera Trajectory (Visual Odometry)...")
    plane_distance = _plane_distance(scene_found, [r for rs in recons for r in rs])
    camera_result = CameraPoseEstimator().estimate_trajectory(
        input_path,
        hand_bboxes=hand_bboxes,
        static_camera=static_camera,
        plane_distance_m=plane_distance,
        K=K,
    )
    if scene_found is not None and scene_found.table is not None:
        camera_result.table_normal = scene_found.table.normal.astype(np.float32)
    camera_poses = camera_result.poses_world_camera
    if len(camera_poses) == 0:
        log.error("Camera reconstruction failure: No camera poses computed.")
        return 1

    if progress_callback:
        progress_callback(68.0, "Stage 2/3: Aligning & Fusing into World Space...")
    fused: dict[str, list] = {}
    for k, side in enumerate(sides):
        if not counts[side]:
            continue
        frames = WorldSpaceFusion(temporal_smoothing=True).fuse_sequence(
            reconstructions=recons[k], camera_poses=camera_poses, fps=fps,
        )
        for f in frames:
            f.hand_side = side
        if frames and not (np.any(np.isnan(frames[0].wrist_world)) or np.any(np.isinf(frames[0].wrist_world))):
            fused[side] = frames

    # The object, held by either hand: every visible pinch point is a candidate.
    object_info, objects = None, None
    try:
        from dfpipeline.retarget.object_tracking import ObjectTracker, lift_to_world

        objects = ObjectTracker()
        object_track = objects.track(input_path, hand_bboxes, pinch_px, fps, progress_callback)
        if object_track is not None:
            by_frame: dict[int, list] = {}
            for frames in fused.values():
                for f in frames:
                    by_frame.setdefault(f.frame_idx, []).append(0.5 * (f.joints_world[4] + f.joints_world[8]))
            pinch_world = {i: np.stack(p) for i, p in by_frame.items()}
            object_info = lift_to_world(
                object_track, camera_poses, K, camera_result.table_normal, plane_distance or 0.5, pinch_world
            )
            _export_object(out_dir, object_track, object_info, fps)
    except Exception as exc:  # noqa: BLE001 - the object is optional, as for one hand
        log.warning("Object tracking skipped: %s", exc)

    if progress_callback:
        progress_callback(74.0, "Stage 2/3: Serializing Trajectory Dataset (JSON & NPZ)...")
    # Each hand's holds, with the handovers between them settled (grasp.py).
    holds = {}
    try:
        from dfpipeline.retarget import grasp

        holds = {side: grasp.find_holds(hand_kp[k], objects.candidates if objects else [], total_frames, fps)
                 for k, side in enumerate(sides)}
        grasp.resolve_handovers(holds, fps)
    except Exception as exc:  # noqa: BLE001 - each hand is tried again on its own below
        log.warning("Holds not found for both hands: %s", exc)
    for side, frames in fused.items():
        metadata = {
            "video_file": input_path.name,
            "fps": fps,
            "total_frames": total_frames,
            "hand": side,
            "reconstructed_frames": len(frames),
            "hand_first_frame": min(f.frame_idx for f in frames),
            "hand_last_frame": max(f.frame_idx for f in frames),
            "camera_reconstruction_source": camera_result.source,
            "hand_model": "MANO-21",
            "coordinate_systems": {
                "camera_frame": "+X: Right, +Y: Down, +Z: Forward (Optical Axis)",
                "world_frame": "+X: Right, +Y: Down, +Z: Forward, relative to initial camera pose at t=0",
            },
        }
        TrajectoryExporter(output_dir=out_dir / side).export(frames, metadata=metadata)
        _write_holds(out_dir / side, hand_kp[sides.index(side)], objects.candidates if objects else [],
                     frames, total_frames, fps, K, holds.get(side))

    _write_hand_stage(out_dir, fps, total_frames, size, camera_result, plane_distance, object_info,
                      hands=list(ARM_SIDES), perception=_perception(lens, up, scene_found))
    log.info(
        "Hand stage (both hands) complete in %.1fs: %s",
        time.perf_counter() - start_time, {s: len(f) for s, f in fused.items()},
    )
    return 0


def _robot_stage_two_arms(hand_dir: Path, out_dir: Path, bundle: dict) -> int:
    """Both hands onto two SO-101 arms, on one clock (see `retarget_pair`)."""
    hands = {}
    for side in bundle["hands"]:
        if not (hand_dir / side).is_dir():
            log.error("The %s hand was never reconstructed; a two-arm trajectory needs both.", side)
            return 1
        hands[side] = _load_hand_frames(hand_dir / side)
    table_normal = np.asarray(bundle["table_normal"], dtype=np.float64) if bundle.get("table_normal") else None
    log.info("Stage 5b: Retargeting both hands to two SO-101 arms: %s",
             {s: len(f) for s, f in hands.items()})
    try:
        from dfpipeline.retarget import grasp

        grips = {side: grasp.read_grip(hand_dir / side) for side in hands}
        arms, frames = retarget_pair(hands, table_normal, bundle.get("table_distance_m") or 0.5, grips)
        export_pair(arms, frames, out_dir)
    except Exception as exc:  # noqa: BLE001 - reported as no trajectory, as for one arm
        log.error("Two-arm SO-101 retargeting failed: %s", exc)
        return 1
    return 0


def run_pipeline(
    input_video: str | Path,
    output_dir: str | Path,
    progress_callback: Any = None,
    static_camera: bool = False,
) -> int:
    """Both stages, one after the other — a whole clip from video to arm.

    The platform runs the two halves separately and at different times. This is
    for running one clip by hand and seeing everything it produces.
    """
    code = run_hand_stage(input_video, output_dir, progress_callback, static_camera)
    if code != 0:
        return code
    return run_robot_stage(output_dir, output_dir)
