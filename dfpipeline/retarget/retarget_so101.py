"""SO-101 Embodiment Retargeting Module.

Converts the world-space human hand trajectory into SO-101 joint trajectories.

Kinematics come from the official URDF (TheRobotStudio/SO-ARM100,
Simulation/SO101/so101_new_calib.urdf), parsed at runtime.

Retargeting (per frame):
1. Human frame -> robot frame. The world frame is the frame-0 camera looking down at
   the table. Robot +Z = up (opposite the table normal), robot +X = the operator's
   average reach direction (wrist -> knuckles) projected onto the table, so the robot
   base sits where the human arm comes from.
2. Target gripper point = midpoint of thumb tip (4) and index tip (8) (the pinch point,
   which corresponds to the point between the SO-101 jaws). Its height above the table
   is metric (camera-to-table distance); its horizontal motion is re-centred into the
   robot's comfortable workspace.
3. Target orientation: approach direction = wrist -> pinch point; jaw roll from the
   thumb -> index direction.
4. Gripper opening from the thumb-index tip distance.
5. Initial pose by multi-start IK; afterwards differential IK (velocity-level bounded least
   squares, the QP formulation used by mink / QP-IK): each frame the joints take a bounded
   step toward the target inside soft joint limits (wrist roll / pan <= +/-95 deg, since a
   parallel-jaw gripper is symmetric under 180 deg roll). The arm moves continuously and can
   never jump to a different IK branch, even across tracking gaps (targets interpolated).
6. Retiming (TOPP-RA idea): the joint path is kept, and time is stretched wherever the servo
   velocity / acceleration limits would be exceeded; the result is resampled at a fixed rate.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dfpipeline.retarget import paths

# Table contact, read from the end effector's height: within CONTACT_BAND_M of its
# lowest point, for at least MIN_CONTACT_S, and not in the first/last EDGE_S (the
# hand entering or leaving the frame at table level).
CONTACT_BAND_M = 0.02
MIN_CONTACT_S = 0.3
EDGE_S = 1.0


def _grasp_window(z: np.ndarray, fps: float, min_contact_s: float | None = None,
                  require_two: bool = False) -> tuple[int | None, int | None]:
    """Grasp = first real table contact, release = last one.

    A contact is a run of frames within CONTACT_BAND_M of the lowest EE height that lasts at
    least MIN_CONTACT_S and does not touch the start/end of the episode (the hand entering or
    leaving the frame at table level is not a grasp).
    """
    low = z < z.min() + CONTACT_BAND_M
    runs, k = [], 0
    while k < len(low):
        if low[k]:
            j = k
            while j + 1 < len(low) and low[j + 1]:
                j += 1
            runs.append((k, j))
            k = j + 1
        else:
            k += 1
    min_len = (MIN_CONTACT_S if min_contact_s is None else min_contact_s) * fps
    edge = int(EDGE_S * fps)
    good = [(a, b) for a, b in runs if (b - a + 1) >= min_len and a > edge and b < len(z) - 1 - edge]
    if require_two:
        return (good[0][0] + int(0.1 * fps), (good[-1][0] + good[-1][1]) // 2) if len(good) >= 2 else (None, None)
    if not good:
        good = [(a, b) for a, b in runs if (b - a + 1) >= min_len] or runs
    if not good:
        return None, None
    close_at = min(good[0][0] + int(0.25 * fps), good[0][1])    # jaws close once the gripper settles
    # Release in the middle of the final table contact (object already set down), leaving the
    # rest of that contact for the fingers to clear before the arm moves away.
    ra, rb = good[-1]
    release_at = max(close_at + 1, (ra + rb) // 2) if len(good) > 1 or rb > close_at else rb
    return close_at, min(release_at, len(z) - 1)

log = logging.getLogger("hand2robot.retarget_so101")

# URDF, scene and calibration are resolved per call: see dfpipeline/retarget/paths.py.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = ARM_JOINTS + ["gripper"]

# Where the human workspace centre is placed in the robot base frame (meters).
ROBOT_WORKSPACE_CENTER = np.array([0.26, 0.0], dtype=np.float64)  # SO-101 shoulder sits ~7 cm forward
CONTACT_HEIGHT_M = 0.02          # gripper-point height at the episode's lowest (contact) moments
MIN_EE_HEIGHT_M = 0.01           # keep the gripper point above the table surface
MIN_REACH_M = 0.16               # keep targets out of the robot's own base (comfortable reach)
MAX_REACH_M = 0.40               # stay inside the SO-101 workspace (max reach ~0.48 m)
MAX_PAN_RAD = np.radians(45)     # keep targets in front of the robot (hand entry/exit at frame edges)
PINCH_CLOSED_M = 0.02            # thumb-index distance treated as fully closed
PINCH_OPEN_M = 0.09              # thumb-index distance treated as fully open

# IK residual weights
W_POS = 1.0
W_APPROACH = 0.015               # 1 rad of approach error ~ 1.5 cm of position error
W_ROLL = 0.01
# Wrist roll is read off the thumb-index line, which is the jaw axis only in a
# pinch. In a whole-hand grasp (a cup) that line wanders, and the roll swung
# limit to limit, 140-190 deg an episode. The line is smoothed over this long,
# and the roll is pulled gently back toward neutral (1 rad ~ 2 cm). On a real
# pick-and-place clip this took the roll's travel from 139 to 28 deg with the
# gripper's position error unchanged (7.7 mm); at 0.05 the roll barely moved.
ROLL_SMOOTH_S = 1.0
W_ROLL_REST = 0.02
W_SMOOTH = 0.02                  # 1 rad of joint change vs previous frame ~ 2 cm: prevents IK flips
W_TABLE = 3.0                    # links below the table clearance are heavily penalised
W_POSTURE = 1.0                  # penalise the upper arm leaning backward behind the shoulder
TABLE_CLEARANCE_M = 0.02         # elbow / wrist / gripper must stay this far above the table
W_NOMINAL = 0.01                 # gentle pull toward a natural reaching pose (1 rad ~ 1 cm)
W_LIMIT = 0.5                    # penalise the last LIMIT_MARGIN_RAD before each joint limit
LIMIT_MARGIN_RAD = np.radians(10)
# Natural table-reaching pose for lift / elbow / wrist_flex (URDF signs: + leans forward / bends down)
NOMINAL_POSE = np.array([0.35, 0.6, 0.6])

# Differential IK (geometric stage)
DIK_ITERATIONS = 4               # bounded least-squares steps per video frame
DIK_DAMPING = 0.02               # Levenberg damping rows (robustness near singularities)
IK_MAX_SPEED_DEG_S = 400.0       # per-frame joint step budget while tracking the path
SOFT_LIMIT_RAD = np.radians(95)  # IK never uses more than +/-95 deg (wrist roll, pan): no spins
STALL_ERROR_M = 0.03             # tracking error that counts as "stuck"
STALL_FRAMES = 5                 # ... for this many frames -> plan a joint-space move to global IK
W_GOAL = 0.5                     # pull toward the planned configuration during that move
# Rest / home pose = LeRobot 0 on every joint, gripper closed (LeRobot gripper 0 = closed).
# LeRobot DEGREES mode reports (reading - (range_min + range_max) / 2) * 360 / 4095, i.e. 0 is
# the midpoint of the range recorded by `lerobot-calibrate` on the user's robot, not the URDF
# zero. With the robot's calibration file, the per-joint offset URDF -> LeRobot is
# (2047 - mid) * 360 / 4095 deg (after half-turn homing the held "middle" pose reads 2047).
# Without the file the offsets are 0 (symmetric-range assumption).

# Rest pose the robot starts from and returns to:
#   "calibration"  = the pose held during `lerobot-calibrate` ("middle of range"), where half-turn
#                    homing makes every motor read 2047 = URDF zero for so101_new_calib (default)
#   "lerobot_zero" = LeRobot 0 on every joint (midpoint of the recorded range)
REST_POSE_MODE = "calibration"
PLACE_SNAP_M = 0.06             # pick/place gaps below this are treated as drift (loop closure)
HALF_TURN_TICKS = 2047.0
TICKS_PER_DEG = 4095.0 / 360.0
HOME_HOLD_S = 1.0                # hold at rest before starting and after returning
HOME_MOVE_DEG_S = 45.0           # nominal speed of the move to/from home (retiming caps it anyway)
W_PAN_ALIGN = 0.05               # pan follows the target azimuth: forbids "reach over the top" flips

# Servo limits used for retiming (conservative for STS3215: ~270 deg/s no-load at 7.4 V).
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
SERVO_MAX_SPEED_DEG_S = [120.0, 100.0, 120.0, 150.0, 150.0]
SERVO_MAX_ACCEL_DEG_S2 = [600.0, 450.0, 600.0, 800.0, 800.0]


def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_angle(axis: np.ndarray, q: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


@dataclass
class _Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray       # 4x4
    axis: np.ndarray         # (3,)
    kind: str
    lower: float
    upper: float


class SO101Kinematics:
    """Forward kinematics of the SO-101 arm from its URDF."""

    def __init__(self, urdf_path: str | Path | None = None) -> None:
        urdf_path = Path(urdf_path) if urdf_path else paths.urdf()
        if not urdf_path.exists():
            raise FileNotFoundError(f"Missing SO-101 URDF: {urdf_path}")
        root = ET.parse(urdf_path).getroot()
        joints: dict[str, _Joint] = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            T = np.eye(4)
            T[:3, :3] = _rpy_matrix(*rpy)
            T[:3, 3] = xyz
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
            lim = j.find("limit")
            joints[j.get("name")] = _Joint(
                name=j.get("name"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                origin=T,
                axis=axis,
                kind=j.get("type"),
                lower=float(lim.get("lower")) if lim is not None else 0.0,
                upper=float(lim.get("upper")) if lim is not None else 0.0,
            )
        self.joints = joints

        # Serial chain base_link -> gripper_frame_link
        by_child = {j.child: j for j in joints.values()}
        chain, link = [], "gripper_frame_link"
        while link in by_child:
            j = by_child[link]
            chain.append(j)
            link = j.parent
        self.chain = list(reversed(chain))
        self.lower = np.array([joints[n].lower for n in ARM_JOINTS])
        self.upper = np.array([joints[n].upper for n in ARM_JOINTS])
        self.gripper_lower = joints["gripper"].lower
        self.gripper_upper = joints["gripper"].upper

    def fk(self, q_arm: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Returns (T_base_gripperframe 4x4, list of 3D points of the chain for drawing)."""
        q = dict(zip(ARM_JOINTS, q_arm))
        T = np.eye(4)
        points = [T[:3, 3].copy()]
        for j in self.chain:
            T = T @ j.origin
            if j.kind == "revolute" and j.name in q:
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j.axis, q[j.name])
                T = T @ R
            points.append(T[:3, 3].copy())
        return T, points


@dataclass
class ArmPath:
    """One arm's solved path, a sample per video frame (`dense_idx`), before the
    home moves and retiming."""

    dense_idx: np.ndarray        # (N,) source video frame of each sample
    q_path: np.ndarray           # (N,5) radians
    tracked: np.ndarray          # (N,) the hand was seen (not interpolated) at this frame
    d_open: np.ndarray           # (N,) gripper opening 0..1
    d_faithful: np.ndarray       # (N,3) the real hand path, robot base frame
    d_obj: np.ndarray            # (N,3) object, NaN where untracked
    d_held: np.ndarray           # (N,) object held
    lo: np.ndarray               # (5,) soft joint limits used
    hi: np.ndarray
    fps: float

    def spanning(self, first: int, last: int) -> ArmPath:
        """The same path over video frames first..last, holding still outside its
        own span: an arm whose hand comes in later waits at its first pose, and
        one whose hand leaves earlier stays at its last."""
        before = max(0, int(self.dense_idx[0]) - first)
        after = max(0, last - int(self.dense_idx[-1]))

        def hold(arr, fill_first=None, fill_last=None):
            arr = np.asarray(arr)
            head = np.repeat(arr[:1], before, 0) if fill_first is None else np.full((before,) + arr.shape[1:], fill_first)
            tail = np.repeat(arr[-1:], after, 0) if fill_last is None else np.full((after,) + arr.shape[1:], fill_last)
            return np.concatenate([head, arr, tail])

        return ArmPath(
            dense_idx=np.arange(first, last + 1),
            q_path=hold(self.q_path), tracked=hold(self.tracked, False, False),
            d_open=hold(self.d_open), d_faithful=hold(self.d_faithful),
            d_obj=hold(self.d_obj), d_held=hold(self.d_held, False, False),
            lo=self.lo, hi=self.hi, fps=self.fps,
        )


@dataclass
class RobotFrame:
    frame_idx: int
    timestamp: float
    q_arm: np.ndarray            # (5,) radians
    gripper: float               # radians
    gripper_open: float          # 0..1
    target_pos: np.ndarray       # (3,) robot base frame
    ee_pos: np.ndarray           # (3,) achieved
    points: list[np.ndarray]     # chain points for drawing
    ee_rot: np.ndarray           # (3,3) gripper frame orientation in the base frame
    position_error: float        # meters
    tracked: bool = True         # False where the target was interpolated across a hand gap
    phase: str = "action"        # home_start / action / home_end
    object_pos: np.ndarray | None = None   # manipulated object base (robot frame), NaN if untracked
    object_held: bool = False


def load_lerobot_offsets(path: str | Path | None = None) -> tuple[np.ndarray, bool, np.ndarray | None]:
    """Per-arm-joint offsets (deg) such that lerobot_deg = urdf_deg + offset, plus the robot's
    real recorded half-range (deg) per joint (LeRobot 0 +/- half_range), or None without a file."""
    path = Path(path) if path else paths.calibration()
    if not path.exists():
        return np.zeros(len(ARM_JOINTS)), False, None
    cal = json.load(open(path, encoding="utf-8"))
    off, half = [], []
    for name in ARM_JOINTS:
        c = cal[name]
        mid = (float(c["range_min"]) + float(c["range_max"])) / 2.0
        off.append((2047.0 - mid) * 360.0 / 4095.0)
        half.append((float(c["range_max"]) - float(c["range_min"])) / 2.0 * 360.0 / 4095.0)
    return np.array(off), True, np.array(half)


class SO101Retargeter:
    """Human hand world trajectory -> SO-101 joint trajectory."""

    def __init__(self, urdf_path: str | Path | None = None) -> None:
        self.kin = SO101Kinematics(urdf_path)
        self.lerobot_offset_deg, self.lerobot_calibrated, self.lerobot_half_range_deg = load_lerobot_offsets()
        self.gripper_range_ticks = None
        # LeRobot reverses a motor whose calibration says drive_mode 1, in the
        # -100..100 reading (not in degrees).
        self.drive_sign = np.ones(len(ARM_JOINTS))
        if self.lerobot_calibrated:
            with open(paths.calibration(), encoding="utf-8") as fh:
                cal = json.load(fh)
            g = cal["gripper"]
            self.gripper_range_ticks = (float(g["range_min"]), float(g["range_max"]))
            self.drive_sign = np.array([-1.0 if int(cal[n].get("drive_mode") or 0) else 1.0 for n in ARM_JOINTS])
        if not self.lerobot_calibrated:
            log.warning(
                "No LeRobot calibration at %s: rest pose assumes symmetric joint ranges. Copy "
                "~/.cache/huggingface/lerobot/calibration/robots/so_follower/<robot_id>.json there "
                "for your robot's exact LeRobot 0.", paths.calibration(),
            )
        # Approach axis and jaw axis of gripper_frame_link, measured from the model.
        self.approach_axis_local = np.array([0.0, 0.0, 1.0])
        self.jaw_axis_local = np.array([1.0, 0.0, 0.0])

    @staticmethod
    def _backward_lean(pts: list[np.ndarray]) -> float:
        """How far (m) the elbow sits behind the shoulder, measured along the arm's reach direction."""
        shoulder, elbow, tip = pts[2], pts[3], pts[-1]
        reach = tip[:2] - shoulder[:2]
        n = np.linalg.norm(reach)
        if n < 1e-6:
            return 0.0
        return float(max(0.0, -((elbow[:2] - shoulder[:2]) @ (reach / n))))

    def _close_pick_place_loop(self, faithful: np.ndarray, frames: list, object_info: dict | None = None) -> np.ndarray:
        """Pick-and-place loop closure: if the object is put down within PLACE_SNAP_M of where
        it was picked up, treat the gap as reconstruction drift (handheld camera odometry /
        depth error while the hand moved toward the camera) and spread the correction linearly
        over the carry, so the place point equals the pick point. Larger moves are kept."""
        self.place_correction_cm = 0.0
        self._loop = None
        fps = self._fps(frames)
        if object_info is not None and object_info.get("picks"):
            # Object-tracked events: only a single pick-and-place cycle is closed (with several
            # cycles the object really is put in different places).
            if len(object_info["picks"]) != 1 or len(object_info["places"]) != 1:
                return faithful
            vid = np.array([f.frame_idx for f in frames])
            pick = int(np.argmin(np.abs(vid - object_info["picks"][0])))
            place = int(np.argmin(np.abs(vid - object_info["places"][0])))
        else:
            # Video-rate hand path: place dips can be short; pick and place must be separate contacts.
            pick, place = _grasp_window(faithful[:, 2], fps, min_contact_s=0.15, require_two=True)
        if pick is None or place is None or place <= pick:
            return faithful
        gap = faithful[pick, :2] - faithful[place, :2]
        if np.linalg.norm(gap) > PLACE_SNAP_M:
            return faithful
        out = faithful.copy()
        w = np.clip((np.arange(len(out)) - pick) / (place - pick), 0.0, 1.0)
        out[:, :2] += w[:, None] * gap
        self.place_correction_cm = float(100 * np.linalg.norm(gap))
        self._loop = (frames[pick].frame_idx, frames[place].frame_idx, gap)
        log.info("Pick-and-place loop closure: corrected %.1f cm of drift between pick and place.",
                 self.place_correction_cm)
        return out

    def _gripper_lerobot(self, grip_rad: float, open_frac: float) -> float:
        """LeRobot gripper.pos (0 = closed .. 100 = open). With the calibration file this is the
        exact RANGE_0_100 mapping of the motor reading (2047 + angle in ticks)."""
        if self.gripper_range_ticks is None:
            return 100.0 * open_frac
        lo_t, hi_t = self.gripper_range_ticks
        raw = HALF_TURN_TICKS + np.degrees(grip_rad) * TICKS_PER_DEG
        return float(np.clip((raw - lo_t) / (hi_t - lo_t) * 100.0, 0.0, 100.0))

    def range_m100(self, q_arm: np.ndarray) -> np.ndarray:
        """Joints 1-5 as LeRobot's SO-101 reports them by default (use_degrees=False):
        -100 .. 100 across each joint's calibrated range, 0 at its middle.

        That is LeRobot degrees over the joint's half range. Without the robot's
        calibration file the range is the URDF's, centred on its zero.
        """
        deg = np.degrees(np.asarray(q_arm, float)) + self.lerobot_offset_deg
        if self.lerobot_half_range_deg is not None:
            half = np.asarray(self.lerobot_half_range_deg, float)
        else:
            half = np.array([max(abs(np.degrees(self.kin.joints[n].lower)), abs(np.degrees(self.kin.joints[n].upper)))
                             for n in ARM_JOINTS])
        return np.clip(self.drive_sign * deg / half * 100.0, -100.0, 100.0)

    @staticmethod
    def _steady_jaw(jaw: np.ndarray, fps: float) -> np.ndarray:
        """The jaw axis with its sign kept continuous and its wander smoothed.

        A parallel jaw is the same turned half a turn, so +axis and -axis are
        one grasp; flipping each frame to the side of the previous one stops
        the smoothing from averaging the two into nothing.
        """
        from scipy.signal import savgol_filter

        jaw = jaw.copy()
        for i in range(1, len(jaw)):
            if jaw[i] @ jaw[i - 1] < 0:
                jaw[i] = -jaw[i]
        window = int(ROLL_SMOOTH_S * fps) | 1
        if len(jaw) > window >= 5:
            jaw = savgol_filter(jaw, window, 2, axis=0)
            jaw /= np.linalg.norm(jaw, axis=1, keepdims=True) + 1e-9
        return jaw

    @staticmethod
    def _best_base_offset(rel: np.ndarray) -> np.ndarray:
        """Where to put the human workspace (median-centred path) relative to the robot base
        so that as much of the real path as possible lies in the comfortable reachable zone."""
        def outside(p):
            r = np.linalg.norm(p, axis=1)
            ang = np.abs(np.arctan2(p[:, 1], p[:, 0]))
            return (np.maximum(0, MIN_REACH_M - r) + np.maximum(0, r - MAX_REACH_M)
                    + r * np.maximum(0, ang - MAX_PAN_RAD))
        best, best_cost = ROBOT_WORKSPACE_CENTER.copy(), np.inf
        for dx in np.arange(0.16, 0.36, 0.01):
            for dy in np.arange(-0.10, 0.101, 0.01):
                c = np.array([dx, dy])
                cost = float(np.sum(outside(rel + c))) + 0.01 * float(np.linalg.norm(c - ROBOT_WORKSPACE_CENTER))
                if cost < best_cost:
                    best, best_cost = c, cost
        return best

    @staticmethod
    def _fps(frames: list) -> float:
        for a, b in zip(frames, frames[1:]):
            if b.frame_idx > a.frame_idx and b.timestamp > a.timestamp:
                return (b.frame_idx - a.frame_idx) / (b.timestamp - a.timestamp)
        return 30.0

    @staticmethod
    def _robot_frame(
        joints_world: np.ndarray,
        table_normal_cam: np.ndarray | None,
    ) -> np.ndarray:
        """3x3 rotation mapping camera-0/world vectors into the robot base frame."""
        # Table normal points from the camera toward the table (n_z > 0). Robot up is opposite.
        n = np.array([0.0, 0.0, 1.0]) if table_normal_cam is None else np.asarray(table_normal_cam, float)
        up = -n / np.linalg.norm(n)
        reach = np.mean(joints_world[:, 9] - joints_world[:, 0], axis=0)
        reach = reach - (reach @ up) * up
        if np.linalg.norm(reach) < 1e-6:
            reach = np.array([0.0, -1.0, 0.0]) - (np.array([0.0, -1.0, 0.0]) @ up) * up
        x = reach / np.linalg.norm(reach)
        y = np.cross(up, x)
        return np.stack([x, y, up])  # rows: robot axes expressed in world coordinates

    def retarget(
        self,
        frames: list,                      # list[WorldHandFrame]
        table_normal_cam: np.ndarray | None,
        table_distance_m: float | None = None,
        object_info: dict | None = None,
        grip: dict | None = None,
    ) -> list[RobotFrame]:
        if not frames:
            return []
        return self.finish([(self, self.solve(frames, table_normal_cam, table_distance_m, object_info, grip))])[0]

    def solve(
        self,
        frames: list,                      # list[WorldHandFrame]
        table_normal_cam: np.ndarray | None,
        table_distance_m: float | None = None,
        object_info: dict | None = None,
        grip: dict | None = None,
    ) -> ArmPath:
        """The joint path, one sample per video frame from the hand's first to its
        last: everything but the home moves and the retiming (`finish`).

        `grip`: the hand's holds and the held object's width (grasp.py). The
        gripper then opens around the object while not holding it and closes
        just inside it while holding — not the thumb-to-index gap, which reads
        a hand wrapped round a cup as open."""
        self.object_info = object_info
        J = np.stack([f.joints_world for f in frames]).astype(np.float64)   # (T,21,3)
        R_rw = self._robot_frame(J, table_normal_cam)
        n = -R_rw[2]                                                         # table normal (toward table)

        pinch = 0.5 * (J[:, 4] + J[:, 8])
        wrist = J[:, 0]
        # Height along the table normal. The absolute camera-to-table distance is only an
        # estimate, so heights are referenced to the episode's lowest pinch positions
        # (grasp / place contact moments), which are robust to that error.
        up_coord = -(pinch @ n)
        heights = up_coord - np.percentile(up_coord, 5) + CONTACT_HEIGHT_M
        # Faithful target = the real hand path, moved rigidly (no scaling / bending), with the
        # robot base placed so as much of that path as possible is naturally reachable.
        rel = pinch @ R_rw[:2].T
        med = np.median(rel, axis=0)
        rel = rel - med                                                      # (T,2) along robot x,y
        base_off = self._best_base_offset(rel)
        faithful_xy = rel + base_off
        z = np.maximum(heights, MIN_EE_HEIGHT_M)
        faithful = np.column_stack([faithful_xy, z])
        faithful = self._close_pick_place_loop(faithful, frames, object_info)
        faithful_xy = faithful[:, :2]

        # Manipulated object in the robot frame, with exactly the hand's mapping (and the same
        # loop-closure correction). Resting on the table -> z = 0; lifted -> height above it.
        obj_robot = obj_held = None
        if object_info is not None:
            ob = np.asarray(object_info["base_world"], float)
            obj_xy = ob @ R_rw[:2].T - med + base_off
            up = -(ob @ n)
            held_v = np.asarray(object_info["held"], bool)
            table_up = np.median(up[~held_v]) if (~held_v).any() else np.min(up)
            obj_z = np.maximum(0.0, up - table_up)
            if self._loop is not None:
                pk, pl, gap = self._loop
                w = np.clip((np.arange(len(ob)) - pk) / max(pl - pk, 1), 0.0, 1.0)
                obj_xy = obj_xy + w[:, None] * gap
            obj_robot = np.column_stack([obj_xy, obj_z])
            obj_held = held_v
        # IK aims at the nearest point inside the comfortable reachable zone.
        r = np.clip(np.linalg.norm(faithful_xy, axis=1), MIN_REACH_M, MAX_REACH_M)
        ang = np.clip(np.arctan2(faithful_xy[:, 1], faithful_xy[:, 0]), -MAX_PAN_RAD, MAX_PAN_RAD)
        targets = np.column_stack([r * np.cos(ang), r * np.sin(ang), z])
        self.reachable_fraction = float(np.mean(np.linalg.norm(targets - faithful, axis=1) < 0.005))

        approach = (pinch - wrist) @ R_rw.T
        approach /= np.linalg.norm(approach, axis=1, keepdims=True) + 1e-9
        jaw = (J[:, 8] - J[:, 4]) @ R_rw.T
        jaw /= np.linalg.norm(jaw, axis=1, keepdims=True) + 1e-9
        jaw = self._steady_jaw(jaw, self._fps(frames))
        aperture = np.linalg.norm(J[:, 8] - J[:, 4], axis=1)
        from dfpipeline.retarget import grasp

        # As the hand opens and closes, shut while it holds (grasp.py).
        opening = grasp.opening(np.array([f.frame_idx for f in frames]), aperture, grip)
        held = bool(grip and grip.get("holds"))
        self.gripper_source = "hand+holds" if held else "hand"
        self.object_width_m = (grip or {}).get("object_width_m") if held else None

        from scipy.optimize import least_squares, lsq_linear
        from scipy.signal import medfilt, savgol_filter

        # Gripper: remove flicker from 2D keypoint noise before mapping to the jaw.
        if len(opening) >= 9:
            opening = np.clip(savgol_filter(medfilt(opening, 5), 9, 2), 0.0, 1.0)

        # Soft joint range used by the IK: URDF limits minus a safety margin, with wrist roll
        # and pan narrowed. The SO-101 gripper is a parallel jaw (symmetric under a 180 deg
        # roll), so wrist roll never needs more than about +/-90 deg: no near-360 spins.
        lo = np.maximum(self.kin.lower + LIMIT_MARGIN_RAD, -SOFT_LIMIT_RAD)
        hi = np.minimum(self.kin.upper - LIMIT_MARGIN_RAD, SOFT_LIMIT_RAD)
        if self.lerobot_half_range_deg is not None:
            # Never command beyond the robot's real recorded range (physical stops), minus margin.
            real_lo = np.radians(-self.lerobot_half_range_deg - self.lerobot_offset_deg) + LIMIT_MARGIN_RAD
            real_hi = np.radians(self.lerobot_half_range_deg - self.lerobot_offset_deg) - LIMIT_MARGIN_RAD
            lo, hi = np.maximum(lo, real_lo), np.minimum(hi, real_hi)

        # ---- Dense target timeline: every video frame from first to last hand frame.
        # A real arm cannot teleport across tracking gaps, so targets are interpolated.
        fps = self._fps(frames)
        self._fps_out = fps
        src_idx = np.array([f.frame_idx for f in frames])
        dense_idx = np.arange(src_idx[0], src_idx[-1] + 1)
        tracked = np.isin(dense_idx, src_idx)

        def interp(arr):
            arr = np.asarray(arr, float)
            flat = arr.reshape(len(arr), -1)
            out = np.stack([np.interp(dense_idx, src_idx, flat[:, k]) for k in range(flat.shape[1])], 1)
            return out.reshape((len(dense_idx),) + arr.shape[1:])

        d_targets = interp(targets)
        d_faithful = interp(faithful)
        d_approach = interp(approach)
        d_approach /= np.linalg.norm(d_approach, axis=1, keepdims=True) + 1e-9
        d_jaw = interp(jaw)
        d_jaw /= np.linalg.norm(d_jaw, axis=1, keepdims=True) + 1e-9
        d_open = np.clip(interp(opening), 0.0, 1.0)
        if obj_robot is not None:
            vi = np.clip(dense_idx, 0, len(obj_robot) - 1)
            d_obj, d_held = obj_robot[vi].copy(), obj_held[vi].copy()
        else:
            d_obj, d_held = np.full((len(dense_idx), 3), np.nan), np.zeros(len(dense_idx), bool)
        if len(d_targets) >= 9:
            d_targets = savgol_filter(d_targets, 9, 2, axis=0)   # EE target noise is amplified by IK
            d_faithful = savgol_filter(d_faithful, 9, 2, axis=0)

        def residual(qa, tgt, app, jw, q_prev=None):
            T, pts = self.kin.fk(qa)
            a = T[:3, :3] @ self.approach_axis_local
            j = T[:3, :3] @ self.jaw_axis_local
            # Parallel jaw: +jw and -jw are the same grasp; follow whichever is closer.
            j_err = j - jw if np.linalg.norm(j - jw) <= np.linalg.norm(j + jw) else j + jw
            parts = [
                W_POS * (T[:3, 3] - tgt),
                W_APPROACH * (a - app),
                W_ROLL * j_err,
                W_TABLE * np.maximum(0.0, TABLE_CLEARANCE_M - np.array([p[2] for p in pts[3:]])),
                [W_POSTURE * self._backward_lean(pts)],
                W_NOMINAL * (qa[1:4] - NOMINAL_POSE),
                [W_ROLL_REST * qa[4]],
                [W_PAN_ALIGN * (qa[0] - np.arctan2(tgt[1], tgt[0]))],
            ]
            if q_prev is not None:
                parts.append(W_SMOOTH * (qa - q_prev))
            return np.concatenate(parts)

        def global_ik(i, q_near=None):
            """Multi-start IK for dense frame i; prefers solutions near q_near (fewer joint moves)."""
            tgt = d_targets[i]
            pan = float(np.clip(np.arctan2(tgt[1], tgt[0]), lo[0], hi[0]))
            starts = [np.array([pan, lift, elbow, wrist, 0.0])
                      for lift in (0.3, 0.9) for elbow in (-0.3, 0.6) for wrist in (0.6, 1.3)]
            if q_near is not None:
                starts.append(q_near)
            best = None
            for q0 in starts:
                sol = least_squares(lambda qa: residual(qa, tgt, d_approach[i], d_jaw[i], q_near),
                                    np.clip(q0, lo, hi), bounds=(lo, hi), max_nfev=200)
                if best is None or sol.cost < best.cost:
                    best = sol
            return best.x

        # ---- 1. Initial pose: global multi-start IK.
        q = global_ik(0)

        # ---- 2. Differential IK (velocity-level, bounded least squares, as in mink / QP-IK):
        # each frame the joints move a bounded step toward the target, never jumping branch.
        step_budget = np.radians(IK_MAX_SPEED_DEG_S) / fps
        eps = 1e-5
        # Stall recovery: differential IK can get trapped near singularities (e.g. arm folded
        # over its base, where pan barely moves the gripper). If tracking stays poor, solve a
        # global IK and move there continuously in joint space (still speed/accel limited).
        q_goal = None
        stall = 0
        q_path = np.zeros((len(dense_idx), 5))
        for i in range(len(dense_idx)):
            tgt, app, jw = d_targets[i], d_approach[i], d_jaw[i]
            T_now, _ = self.kin.fk(q)
            stall = stall + 1 if np.linalg.norm(T_now[:3, 3] - tgt) > STALL_ERROR_M else 0
            if q_goal is None and stall >= STALL_FRAMES:
                cand = global_ik(i, q)
                T_c, _ = self.kin.fk(cand)
                if np.linalg.norm(T_c[:3, 3] - tgt) < np.linalg.norm(T_now[:3, 3] - tgt) - 0.01:
                    q_goal = cand
                stall = 0
            if q_goal is not None and np.max(np.abs(q - q_goal)) < np.radians(3):
                q_goal = None

            def res(qa):
                r = residual(qa, tgt, app, jw)
                return r if q_goal is None else np.concatenate([r, W_GOAL * (qa - q_goal)])

            q_start = q.copy()
            for _ in range(DIK_ITERATIONS):
                r0 = res(q)
                Jr = np.empty((len(r0), 5))
                for k in range(5):
                    dq = np.zeros(5)
                    dq[k] = eps
                    Jr[:, k] = (res(q + dq) - r0) / eps
                # Damped (Levenberg) rows keep steps small near singularities.
                A = np.vstack([Jr, DIK_DAMPING * np.eye(5)])
                b = np.concatenate([-r0, np.zeros(5)])
                moved = q - q_start
                lb = np.maximum(lo - q, -step_budget - moved)
                ub = np.minimum(hi - q, step_budget - moved)
                lb, ub = np.minimum(lb, 0.0), np.maximum(ub, 0.0)
                sol = lsq_linear(A, b, bounds=(lb, ub), method="bvls")
                q = np.clip(q + sol.x, lo, hi)
                if np.linalg.norm(sol.x) < 1e-5:
                    break
            q_path[i] = q

        if len(q_path) >= 15:
            q_path = np.clip(savgol_filter(q_path, 15, 3, axis=0), lo, hi)

        return ArmPath(
            dense_idx=dense_idx, q_path=q_path, tracked=tracked, d_open=d_open,
            d_faithful=d_faithful, d_obj=d_obj, d_held=d_held, lo=lo, hi=hi, fps=fps,
        )

    @staticmethod
    def finish(arms: list[tuple[SO101Retargeter, ArmPath]]) -> list[list[RobotFrame]]:
        """Home moves, one retiming, and the fixed-rate samples — for one arm or several.

        Arms working together share one clock: their paths are retimed as one,
        so a sample is the same instant on every arm, and the slowest joint of
        any arm sets the pace. Their paths must cover the same video frames
        (see `align`). With one arm this is exactly that arm's retiming.
        """
        fps = arms[0][1].fps
        k = len(arms)

        # ---- 2b. Home pose: hold, ease into the first action pose, and after the action
        # ease back and hold. Cosine (smooth start/stop) profiles; retiming below still
        # enforces the servo speed / acceleration limits on these moves. Arms whose
        # moves are shorter wait at home, so all start and finish their moves together.
        def ease(a, b):
            n = max(2, int(np.ceil(np.max(np.abs(b - a)) / np.radians(HOME_MOVE_DEG_S) * fps)) + 1)
            s_ = (1 - np.cos(np.linspace(0.0, np.pi, n))) / 2
            return a + s_[:, None] * (b - a)

        homes, pres, posts = [], [], []
        for r, path in arms:
            # Rest pose: calibration pose (URDF zero = raw 2047) or LeRobot 0 (URDF = -offset).
            rest = np.zeros(5) if REST_POSE_MODE == "calibration" else -np.radians(r.lerobot_offset_deg)
            home = np.clip(rest, path.lo, path.hi)
            hold = np.repeat(home[None], max(1, int(HOME_HOLD_S * fps)), axis=0)
            homes.append(home)
            pres.append(np.vstack([hold, ease(home, path.q_path[0])[1:-1]]))
            posts.append(np.vstack([ease(path.q_path[-1], home)[1:-1], hold]))
        n_pre, n_post = max(len(x) for x in pres), max(len(x) for x in posts)
        n_action = len(arms[0][1].dense_idx)

        def pad(arr):
            arr = np.asarray(arr)
            return np.concatenate([np.repeat(arr[:1], n_pre, 0), arr, np.repeat(arr[-1:], n_post, 0)])

        dense_idx = pad(arms[0][1].dense_idx)
        phase = np.array(["home_start"] * n_pre + ["action"] * n_action + ["home_end"] * n_post)
        per_arm = []
        for (r, path), home, pre, post in zip(arms, homes, pres, posts):
            pre = np.vstack([np.repeat(home[None], n_pre - len(pre), 0), pre])
            post = np.vstack([post, np.repeat(home[None], n_post - len(post), 0)])
            q_path = np.vstack([pre, path.q_path, post])
            tracked = np.concatenate([np.zeros(n_pre, bool), path.tracked, np.zeros(n_post, bool)])
            # Gripper at rest: calibration pose (URDF 0 rad) or closed for "lerobot_zero".
            rest_grip = 0.0 if REST_POSE_MODE == "calibration" else r.kin.gripper_lower
            home_open = (rest_grip - r.kin.gripper_lower) / (r.kin.gripper_upper - r.kin.gripper_lower)
            d_open = np.concatenate([np.full(n_pre, home_open), path.d_open, np.full(n_post, home_open)])
            d_faithful = pad(path.d_faithful)
            d_obj = pad(path.d_obj)
            d_held = np.concatenate([np.zeros(n_pre, bool), path.d_held, np.zeros(n_post, bool)])
            # During home moves the "target" is the arm's own gripper point (no tracking error).
            for j in list(range(n_pre)) + list(range(n_pre + n_action, len(q_path))):
                T_j, _ = r.kin.fk(q_path[j])
                d_faithful[j] = T_j[:3, 3]
            per_arm.append((q_path, tracked, d_open, d_faithful, d_obj, d_held))

        # ---- 3. Retiming (TOPP-style): keep the joint path, stretch time wherever the
        # SO-101 servos' velocity / acceleration limits would be exceeded — on any arm.
        q_all = np.hstack([a[0] for a in per_arm])                     # (N, 5k)
        vmax = np.tile(np.radians(np.array(SERVO_MAX_SPEED_DEG_S)), k)
        amax = np.tile(np.radians(np.array(SERVO_MAX_ACCEL_DEG_S2)), k)
        robot_t, t_out, q_out, excess = retime(q_all, fps, vmax, amax)
        video_t = (dense_idx - dense_idx[0]) / fps
        # The slowdown is the action's: the moves to and from home have no
        # video behind them and are not in the dataset. Counted in, a short
        # clip read as 27x when the hand's own motion was barely slowed.
        a0, a1 = n_pre, n_pre + n_action - 1
        action_robot_s = float(robot_t[a1] - robot_t[a0])
        action_video_s = float(video_t[a1] - video_t[a0])
        for r, _ in arms:
            r.retime_stats = {
                "video_duration_s": round(action_video_s, 3),
                "robot_duration_s": round(action_robot_s, 3),
                "slowdown": round(action_robot_s / max(action_video_s, 1e-6), 3) if action_video_s > 0 else 1.0,
                "peak_speed_or_accel_vs_limit": round(excess, 3),
                "hand_path_reachable_fraction": round(r.reachable_fraction, 3),
                "pick_place_loop_closure_cm": round(getattr(r, "place_correction_cm", 0.0), 2),
            }

        # Resampled at the video fps on the robot clock (fixed rate for LeRobot datasets).
        n_out = len(t_out)
        src_out = np.interp(t_out, robot_t, dense_idx.astype(float))
        knot = np.clip(np.searchsorted(robot_t, t_out, side="right") - 1, 0, len(phase) - 1)
        phase_out = phase[knot]

        results: list[list[RobotFrame]] = []
        for a, ((r, _), (_q, tracked, d_open, d_faithful, d_obj, d_held)) in enumerate(zip(arms, per_arm)):
            q_arm_out = q_out[:, 5 * a:5 * a + 5]
            open_out = np.interp(t_out, robot_t, d_open)
            # Exported target = the faithful (real hand) path; error is measured against it.
            tgt_out = np.stack([np.interp(t_out, robot_t, d_faithful[:, c]) for c in range(3)], 1)
            tracked_out = np.interp(t_out, robot_t, tracked.astype(float)) > 0.5
            obj_out = np.stack([np.interp(t_out, robot_t, d_obj[:, c]) for c in range(3)], 1)
            held_out = np.interp(t_out, robot_t, d_held.astype(float)) > 0.5

            out: list[RobotFrame] = []
            for i in range(n_out):
                T, pts = r.kin.fk(q_arm_out[i])
                g = r.kin.gripper_lower + open_out[i] * (r.kin.gripper_upper - r.kin.gripper_lower)
                out.append(RobotFrame(
                    frame_idx=int(round(src_out[i])),
                    timestamp=float(t_out[i]),
                    q_arm=q_arm_out[i].copy(),
                    gripper=float(g),
                    gripper_open=float(open_out[i]),
                    target_pos=tgt_out[i].copy(),
                    ee_pos=T[:3, 3].copy(),
                    points=pts,
                    ee_rot=T[:3, :3].copy(),
                    position_error=float(np.linalg.norm(T[:3, 3] - tgt_out[i])),
                    tracked=bool(tracked_out[i]),
                    phase=str(phase_out[i]),
                    object_pos=obj_out[i].copy(),
                    object_held=bool(held_out[i]),
                ))

            err = np.array([f.position_error for f in out])
            log.info(
                "SO-101 retargeting: %d robot frames | error median %.1f mm, p95 %.1f mm | "
                "robot time %.1fs vs video %.1fs (x%.2f)",
                len(out), 1000 * np.median(err), 1000 * np.percentile(err, 95),
                action_robot_s, action_video_s, r.retime_stats["slowdown"],
            )
            results.append(out)
        return results

    def record(self, r: RobotFrame) -> dict:
        """One sample as written to so101_trajectory.json."""
        # LeRobot degrees = URDF degrees + calibration offset (0 at rest).
        deg = np.degrees(r.q_arm) + self.lerobot_offset_deg
        action = {f"{n}.pos": round(float(v), 3) for n, v in zip(ARM_JOINTS, deg)}
        action["gripper.pos"] = round(self._gripper_lerobot(r.gripper, r.gripper_open), 2)
        return {
            "frame": int(r.frame_idx),
            "timestamp": round(float(r.timestamp), 4),
            "joints_rad": [round(float(v), 5) for v in r.q_arm] + [round(r.gripper, 5)],
            "action": action,
            "target_pos": [round(float(v), 4) for v in r.target_pos],
            "ee_pos": [round(float(v), 4) for v in r.ee_pos],
            "position_error_m": round(r.position_error, 4),
            "chain_points": [[round(float(c), 4) for c in p] for p in r.points],
            "ee_axes": {
                "approach": [round(float(v), 4) for v in r.ee_rot @ self.approach_axis_local],
                "jaw": [round(float(v), 4) for v in r.ee_rot @ self.jaw_axis_local],
            },
            "gripper_open": round(r.gripper_open, 3),
            "tracked": bool(r.tracked),
            "phase": r.phase,
            **({"object_pos": [round(float(v), 4) for v in r.object_pos], "object_held": bool(r.object_held)}
               if r.object_pos is not None and not np.isnan(r.object_pos[0]) else {}),
        }

    def header(self) -> dict:
        """Everything in so101_trajectory.json but the samples."""
        return {
            "robot": "SO-101 (so101_new_calib.urdf)",
            "joint_names": ALL_JOINTS,
            "joint_limits_deg": {
                n: [round(float(np.degrees(self.kin.joints[n].lower)), 1),
                    round(float(np.degrees(self.kin.joints[n].upper)), 1)]
                for n in ALL_JOINTS
            },
            "units": {"joints_rad": "radians", "action": "degrees (arm), 0-100 (gripper)",
                      "positions": "meters, robot base frame (+X forward, +Z up)",
                      "timestamp": "seconds on the robot clock (retimed); frame = source video frame"},
            "fps": round(float(self._fps_out), 4),
            # Where the gripper's open and close came from: the hand's holds
            # and the object's width (grasp.py), or the thumb-to-index gap.
            "gripper": {"source": getattr(self, "gripper_source", "fingertips"),
                        "object_width_m": getattr(self, "object_width_m", None),
                        "max_gap_m": _max_gap()},
            "retiming": getattr(self, "retime_stats", {}),
            **({"object": {k: self.object_info[k] for k in
                           ("label", "diameter_m", "height_m_prior", "mass_kg_estimate")}}
               if getattr(self, "object_info", None) else {}),
            "lerobot_calibration": {
                "file_found": bool(self.lerobot_calibrated),
                "file": str(paths.calibration()),
                "offset_deg_urdf_to_lerobot": dict(zip(ARM_JOINTS, [round(float(v), 3) for v in self.lerobot_offset_deg])),
                "real_range_deg_lerobot": (dict(zip(ARM_JOINTS, [[round(-float(h), 1), round(float(h), 1)] for h in self.lerobot_half_range_deg]))
                                           if self.lerobot_half_range_deg is not None else None),
                "action_units": "LeRobot degrees (SO101FollowerConfig use_degrees=True); gripper.pos 0 = closed",
                "rest_pose": REST_POSE_MODE,
            },
            "motor_limits": {
                "max_speed_deg_s": dict(zip(ARM_JOINTS, SERVO_MAX_SPEED_DEG_S)),
                "max_accel_deg_s2": dict(zip(ARM_JOINTS, SERVO_MAX_ACCEL_DEG_S2)),
                "soft_range_deg": round(float(np.degrees(SOFT_LIMIT_RAD)), 1),
            },
        }

    def export(self, robot_frames: list[RobotFrame], output_dir: str | Path) -> tuple[Path, Path]:
        """Write so101_trajectory.json (UI + LeRobot-style names) and so101_trajectory.npz."""
        output_dir = Path(output_dir)
        records = [self.record(r) for r in robot_frames]
        payload = {
            **self.header(),
            "trajectory": records,
        }
        json_path = output_dir / "so101_trajectory.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        npz_path = output_dir / "so101_trajectory.npz"
        np.savez_compressed(
            npz_path,
            frames=np.array([r.frame_idx for r in robot_frames], dtype=np.int32),
            timestamps=np.array([r.timestamp for r in robot_frames], dtype=np.float32),
            joints_rad=np.array([np.append(r.q_arm, r.gripper) for r in robot_frames], dtype=np.float32),
            gripper_open=np.array([r.gripper_open for r in robot_frames], dtype=np.float32),
            ee_pos=np.array([r.ee_pos for r in robot_frames], dtype=np.float32),
            target_pos=np.array([r.target_pos for r in robot_frames], dtype=np.float32),
            joint_names=np.array(ALL_JOINTS),
        )
        log.info("Exported SO-101 trajectory to %s and %s", json_path, npz_path)
        return json_path, npz_path


def retime(q_all: np.ndarray, fps: float, vmax: np.ndarray, amax: np.ndarray):
    """Retiming (TOPP-style): keep the joint path, stretch time wherever the
    joints' velocity / acceleration limits would be exceeded, then sample it
    at the video rate on the robot's clock. Shared by every arm (SO-101 and
    any URDF robot). Returns (robot_t, t_out, q_out, peak excess vs limits)."""
    from scipy.interpolate import PchipInterpolator

    dq = np.diff(q_all, axis=0)

    def once(v_lim, a_lim):
        dt = np.maximum(1.0 / fps, np.max(np.abs(dq) / v_lim, axis=1))  # velocity limit
        for _ in range(2000):                                      # acceleration limit
            changed = False
            for j in list(range(1, len(dt))) + list(range(len(dt) - 1, 0, -1)):
                v0, v1 = dq[j - 1] / dt[j - 1], dq[j] / dt[j]
                excess = np.max(np.abs(v1 - v0) / (a_lim * 0.5 * (dt[j - 1] + dt[j])))
                if excess > 1.0 + 1e-6:
                    i = j if np.max(np.abs(v1)) >= np.max(np.abs(v0)) else j - 1
                    dt[i] *= min(1.5, 1.0 + 0.5 * (excess - 1.0) + 0.02)
                    changed = True
            if not changed:
                break
        robot_t = np.concatenate([[0.0], np.cumsum(dt)])
        n_out = int(np.floor(robot_t[-1] * fps)) + 1
        t_out = np.arange(n_out) / fps
        # Shape-preserving cubic (PCHIP): smooth velocity at the knots, no overshoot.
        q_out = PchipInterpolator(robot_t, q_all, axis=0)(t_out)
        return robot_t, t_out, q_out

    def measured_excess(q_out):
        if len(q_out) < 3:
            return 1.0
        v = np.diff(q_out, axis=0) * fps
        acc = np.diff(v, axis=0) * fps
        return max(float(np.max(np.abs(v) / vmax)), float(np.max(np.abs(acc) / amax)))

    # Verify on the final fixed-rate samples (what the motors will actually
    # receive); tighten the internal limits until the sampled trajectory is
    # within the motor limits.
    scale = 1.0
    for _ in range(8):
        robot_t, t_out, q_out = once(vmax * scale, amax * scale)
        excess = measured_excess(q_out)
        if excess <= 1.0:
            break
        scale /= excess * 1.05
    return robot_t, t_out, q_out, excess


# ---- Both hands: two SO-101 arms, one per hand ------------------------------------------

#: The arms of a two-hand clip, in the order their joints are written.
ARM_SIDES = ("left", "right")


def _max_gap() -> float:
    from dfpipeline.config import settings

    return settings.retarget_gripper_max_gap_m


def retarget_pair(
    hands: dict[str, list],
    table_normal_cam: np.ndarray | None,
    table_distance_m: float | None = None,
    grips: dict[str, dict | None] | None = None,
) -> tuple[dict[str, SO101Retargeter], dict[str, list[RobotFrame]]]:
    """Each hand onto its own SO-101, the two on one shared clock.

    Each arm is solved exactly as a lone arm would be, from its own hand and
    with its own base placed where that hand works. Their paths are then laid
    over the same video frames — an arm whose hand comes in later waits at its
    first pose — and retimed together, so a sample is the same instant on both
    and the handover lines up.

    No object is passed in. Its grasp windows belong to whichever hand holds it
    at the time, and a handover is both; each gripper follows its own hand's
    fingers instead.
    """
    sides = [s for s in ARM_SIDES if hands.get(s)]
    if len(sides) != 2:
        raise ValueError(f"both hands are needed, got {sides or 'none'}")
    arms = {s: SO101Retargeter() for s in sides}
    solved = {s: arms[s].solve(hands[s], table_normal_cam, table_distance_m, None, (grips or {}).get(s))
              for s in sides}
    first = min(int(p.dense_idx[0]) for p in solved.values())
    last = max(int(p.dense_idx[-1]) for p in solved.values())
    done = SO101Retargeter.finish([(arms[s], solved[s].spanning(first, last)) for s in sides])
    return arms, dict(zip(sides, done))


def export_pair(
    arms: dict[str, SO101Retargeter],
    frames: dict[str, list[RobotFrame]],
    output_dir: str | Path,
) -> Path:
    """so101_trajectory.json for two arms.

    One sample per instant of the shared clock. `action` holds both arms as a
    two-arm SO-101 dataset records them (`left_arm_shoulder_pan.pos` …
    `right_arm_gripper.pos`, as in e.g. Qiu-Xinchuan/so101_towel): joints 1-5
    in -100 .. 100 of their calibrated range (`range_m100`), the gripper 0 .. 100.
    Each arm's own sample — the one a single arm would write, in degrees — is
    under its side. The headline numbers are the worse arm's.
    """
    output_dir = Path(output_dir)
    sides = list(frames)
    lead = arms[sides[0]]
    records = []
    for samples in zip(*(frames[s] for s in sides)):
        own = {s: arms[s].record(r) for s, r in zip(sides, samples)}
        action = {}
        for s, r in zip(sides, samples):
            for name, value in zip(ARM_JOINTS, arms[s].range_m100(r.q_arm)):
                action[f"{s}_arm_{name}.pos"] = round(float(value), 3)
            action[f"{s}_arm_gripper.pos"] = own[s]["action"]["gripper.pos"]
        records.append({
            "frame": own[sides[0]]["frame"],
            "timestamp": own[sides[0]]["timestamp"],
            "phase": own[sides[0]]["phase"],
            "action": action,
            "position_error_m": max(own[s]["position_error_m"] for s in sides),
            "tracked": all(own[s]["tracked"] for s in sides),
            **own,
        })
    retiming = dict(lead.retime_stats)
    retiming["hand_path_reachable_fraction"] = min(
        arms[s].retime_stats["hand_path_reachable_fraction"] for s in sides
    )
    payload = {
        **lead.header(),
        "robot": "Two SO-101 arms (so101_new_calib.urdf), left and right",
        "arms": sides,
        "joint_names": [f"{s}_arm_{n}" for s in sides for n in ALL_JOINTS],
        "action_units": "joints 1-5: -100..100 of the calibrated range (SOFollower use_degrees=False); "
                        "gripper 0..100, 0 closed",
        "retiming": retiming,
        "retiming_per_arm": {s: arms[s].retime_stats for s in sides},
        "trajectory": records,
    }
    json_path = output_dir / "so101_trajectory.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    log.info("Exported two-arm SO-101 trajectory to %s", json_path)
    return json_path
