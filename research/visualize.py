"""Videos of what the pipeline made of a clip, to look at (research only).

For one clip run through the pipeline (research/hand2robot2_colab.ipynb):

  hand_overlay.mp4   the clip with the tracked hand drawn on it, and how its
                     distances were made (lens, table, hand scale)
  robot_sim.mp4      the SO-101 in MuJoCo replaying the trajectory, on the
                     robot's own clock; the hand's target as a red dot
  two_view.mp4       both, side by side, on the video's clock
  compare.mp4        (compare()) one config's two-view above another's

Not part of the pipeline, never deployed: the platform keeps no drawn video
(dfpipeline/retarget/__init__.py). MuJoCo needs the SO-101 model with its
meshes (TheRobotStudio/SO-ARM100, Simulation/SO101: so101_new_calib.xml is
the same robot as models/so101/so101_new_calib.urdf); without it the arm is
drawn as a stick figure from the trajectory's own link points.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# The hand's bones, in MediaPipe's / MANO's 21-joint order.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
COLOURS = {"left": (255, 140, 0), "right": (0, 200, 255), "": (0, 230, 120)}   # BGR
SIM_SIZE = (480, 480)       # width, height of each robot view
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


# ------------------------------------------------------------------ reading

def _hands(hand_dir: Path) -> dict[str, dict[int, np.ndarray]]:
    """side -> frame -> joints in camera axes (21, 3); one hand is side ""."""
    folders = [(s, hand_dir / s) for s in ("left", "right") if (hand_dir / s / "trajectory.npz").is_file()]
    if not folders and (hand_dir / "trajectory.npz").is_file():
        folders = [("", hand_dir)]
    out = {}
    for side, folder in folders:
        data = np.load(folder / "trajectory.npz")
        out[side] = {int(f): np.asarray(j, float) for f, j in zip(data["frames"], data["joints_camera"])}
    return out


def _K(bundle: dict, width: int, height: int) -> np.ndarray:
    """The lens the clip was reconstructed with (its field of view is in the bundle)."""
    hfov = (bundle.get("perception") or {}).get("hfov_deg")
    f = (width / 2.0) / math.tan(math.radians(hfov) / 2.0) if hfov else 0.95 * max(width, height)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def _robot(robot_dir: Path) -> tuple[dict, list[dict]] | None:
    path = robot_dir / "so101_trajectory.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return payload, payload.get("trajectory") or []


def _arms(record: dict, sides: list[str]) -> dict[str, dict]:
    """side -> that arm's own sample (one arm: side "")."""
    if sides:
        return {s: record[s] for s in sides if s in record}
    return {"": record}


# ------------------------------------------------------------------ drawing

def _banner(img, lines: list[str]) -> None:
    import cv2

    y = 22
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def _draw_hand(img, joints: np.ndarray, K: np.ndarray, colour) -> None:
    import cv2

    z = np.clip(joints[:, 2:3], 1e-3, None)
    px = (joints[:, :2] / z) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    pts = [tuple(int(round(v)) for v in p) for p in px]
    for a, b in BONES:
        cv2.line(img, pts[a], pts[b], colour, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
    # How far the hand is: the pinch point's depth, beside it.
    pinch = 0.5 * (joints[4] + joints[8])
    cv2.putText(img, f"{pinch[2]:.2f} m", (pts[8][0] + 8, pts[8][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                cv2.LINE_AA)


class _Sim:
    """The SO-101 in MuJoCo, one renderer, posed per sample."""

    def __init__(self, so101_dir: Path) -> None:
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(so101_dir / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, SIM_SIZE[1], SIM_SIZE[0])
        self.qadr = {n: self.model.jnt_qposadr[self.model.joint(n).id] for n in ARM_JOINTS}
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.18, 0.0, 0.12]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 0.75, 145.0, -25.0

    def frame(self, sample: dict):
        q = sample.get("joints_rad") or []
        for name, value in zip(ARM_JOINTS, q):
            self.data.qpos[self.qadr[name]] = value
        self.mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        target = sample.get("target_pos")
        if target is not None:
            scene = self.renderer.scene
            if scene.ngeom < scene.maxgeom:
                self.mj.mjv_initGeom(scene.geoms[scene.ngeom], self.mj.mjtGeom.mjGEOM_SPHERE,
                                     np.array([0.012, 0, 0]), np.asarray(target, float), np.eye(3).ravel(),
                                     np.array([1, 0.1, 0.1, 0.9], dtype=np.float32))
                scene.ngeom += 1
        return self.renderer.render()[:, :, ::-1].copy()   # RGB -> BGR

    def close(self) -> None:
        self.renderer.close()


def _stick(sample: dict):
    """The arm from its link points, seen from the side, when MuJoCo is not there."""
    import cv2

    img = np.full((SIM_SIZE[1], SIM_SIZE[0], 3), 40, np.uint8)
    pts = np.asarray(sample.get("chain_points") or [], float)
    if len(pts):
        scale, origin = 900.0, np.array([SIM_SIZE[0] * 0.3, SIM_SIZE[1] * 0.8])
        px = [tuple(int(v) for v in origin + scale * np.array([p[0], -p[2]])) for p in pts]
        for a, b in zip(px, px[1:]):
            cv2.line(img, a, b, (220, 220, 220), 6, cv2.LINE_AA)
        if sample.get("target_pos") is not None:
            t = np.asarray(sample["target_pos"], float)
            cv2.circle(img, tuple(int(v) for v in origin + scale * np.array([t[0], -t[2]])), 6, (60, 60, 255), -1)
    cv2.putText(img, "stick view (no MuJoCo model)", (10, SIM_SIZE[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)
    return img


class _Writer:
    """H.264 MP4 (plays in a browser and in Google Drive's preview)."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        import av

        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=round(fps))
        self.stream.width, self.stream.height, self.stream.pix_fmt = width - width % 2, height - height % 2, "yuv420p"
        self.av = av

    def write(self, bgr) -> None:
        frame = self.av.VideoFrame.from_ndarray(bgr[: self.stream.height, : self.stream.width], format="bgr24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def _fit(img, height: int):
    import cv2

    h, w = img.shape[:2]
    return cv2.resize(img, (int(round(w * height / h)), height))


# ------------------------------------------------------------------ the videos

def render(video: Path, run_dir: Path, title: str, so101_dir: Path | None = None,
           out_dir: Path | None = None) -> dict[str, Path]:
    """The three videos for one clip and one config. `run_dir` holds `hand/`
    and `robot/` as the notebook writes them. Returns what was written."""
    import cv2

    run_dir, out_dir = Path(run_dir), Path(out_dir or Path(run_dir) / "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((run_dir / "hand" / "hand_stage.json").read_text())
    p = bundle.get("perception") or {}
    hands = _hands(run_dir / "hand")
    robot = _robot(run_dir / "robot")
    header, samples = robot if robot else ({}, [])
    sides = header.get("arms") or []

    sim = None
    if so101_dir and (Path(so101_dir) / "scene.xml").is_file():
        try:
            sim = _Sim(Path(so101_dir))
        except Exception as exc:  # noqa: BLE001 - the stick view stands in
            print("MuJoCo not available, drawing a stick view:", exc)

    def robot_view(record: dict):
        views = []
        for side, arm in _arms(record, sides).items():
            img = sim.frame(arm) if sim else _stick(arm)
            if side:   # at the bottom: the banner is at the top
                cv2.putText(img, f"{side} arm", (10, SIM_SIZE[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            COLOURS[side], 2, cv2.LINE_AA)
            views.append(img)
        return np.hstack(views) if views else np.zeros((SIM_SIZE[1], SIM_SIZE[0], 3), np.uint8)

    lines = [
        title,
        f"lens: {p.get('lens_source', 'guess')} {p.get('hfov_deg', '?')} deg | hand: {p.get('hand_model', 'mediapipe')}",
        f"table: {bundle.get('table_distance_m') or 0:.2f} m ({p.get('table_source') or 'from hand'})"
        f" | hand scale: {p.get('hand_scale') or '-'} (measured {p.get('hand_scale_measured') or '-'})",
    ]
    retiming = header.get("retiming") or {}
    if retiming:
        lines.append(f"robot: reach {retiming.get('hand_path_reachable_fraction', '?')}"
                     f" | slowdown {retiming.get('slowdown', '?')}x")

    written = {}
    # 1. the robot alone, on its own clock
    if samples:
        first = robot_view(samples[0])
        w = _Writer(out_dir / "robot_sim.mp4", float(header.get("fps") or 30), first.shape[1], first.shape[0])
        for s in samples:
            img = robot_view(s)
            _banner(img, [title, f"phase: {s.get('phase')}  error: {100 * (s.get('position_error_m') or 0):.1f} cm"])
            w.write(img)
        w.close()
        written["robot_sim"] = out_dir / "robot_sim.mp4"

    # 2. the hand on the clip, and 3. both side by side, on the video's clock
    by_frame = {}
    for s in samples:
        if s.get("phase") == "action":
            by_frame[int(s["frame"])] = s
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps if 5 <= fps <= 65 else 30.0
    hand_w = two_w = None
    last = samples[0] if samples else None
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        K = _K(bundle, img.shape[1], img.shape[0])
        for side, frames in hands.items():
            if idx in frames:
                _draw_hand(img, frames[idx], K, COLOURS.get(side, COLOURS[""]))
        _banner(img, lines)
        if hand_w is None:
            hand_w = _Writer(out_dir / "hand_overlay.mp4", fps, img.shape[1], img.shape[0])
        hand_w.write(img)
        if samples:
            last = by_frame.get(idx, last)
            right = robot_view(last)
            left = _fit(img, right.shape[0])
            both = np.hstack([left, right])
            if two_w is None:
                two_w = _Writer(out_dir / "two_view.mp4", fps, both.shape[1], both.shape[0])
            two_w.write(both)
        idx += 1
    cap.release()
    for w, name in ((hand_w, "hand_overlay"), (two_w, "two_view")):
        if w is not None:
            w.close()
            written[name] = out_dir / f"{name}.mp4"
    if sim:
        sim.close()
    return written


def compare(top: Path, bottom: Path, dest: Path, height: int = 360) -> Path | None:
    """Two two-view videos stacked: one config above another, frame by frame.
    None when either could not be read (nothing is written)."""
    import cv2

    a, b = cv2.VideoCapture(str(top)), cv2.VideoCapture(str(bottom))
    if not (a.isOpened() and b.isOpened()):
        return None
    fps = a.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    while True:
        ok1, x = a.read()
        ok2, y = b.read()
        if not (ok1 and ok2):
            break
        x, y = _fit(x, height), _fit(y, height)
        width = max(x.shape[1], y.shape[1])
        x = cv2.copyMakeBorder(x, 0, 0, 0, width - x.shape[1], cv2.BORDER_CONSTANT)
        y = cv2.copyMakeBorder(y, 0, 0, 0, width - y.shape[1], cv2.BORDER_CONSTANT)
        frame = np.vstack([x, y])
        if writer is None:
            writer = _Writer(dest, fps, frame.shape[1], frame.shape[0])
        writer.write(frame)
    a.release()
    b.release()
    if writer is None:
        return None
    writer.close()
    return dest
