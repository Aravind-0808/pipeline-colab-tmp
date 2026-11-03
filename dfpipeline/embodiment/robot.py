"""A robot as its URDF says it is: the chain from base to end-effector, forward
kinematics, and inverse kinematics within its joint limits.

Built from a URDF and the profile confirmed when the robot was imported (the
app's robot/urdf.py): base and end-effector links, the arm joints in order,
the gripper. Any serial arm — 5, 6 or 7 joints — goes through the same code;
nothing here knows which robot it is.

IK is a damped least-squares fit (scipy), one frame at a time, seeded from the
frame before: position first, the gripper's approach direction second, a
gentle pull towards home last — so a 5-joint arm that cannot match an
orientation still lands on the point, and a 7-joint arm does not wander.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

ACTUATED = {"revolute", "continuous", "prismatic"}
#: A shoulder this close above the base means the arm stands on the table
#: (SO-101, YAM, Piper), its base on the table's surface; else it hangs from
#: a torso (OpenArm) and the table is somewhere below.
TABLE_MOUNTED_SHOULDER_M = 0.25
#: Following a hand (ik(soft=True)), the IK keeps this far inside each joint limit — or this share of a small
#: range: the stop is where joints park and the next frame cannot move.
LIMIT_MARGIN_RAD = np.radians(10)
LIMIT_MARGIN_FRACTION = 0.1
#: Following a hand, the gripper's roll joint stays within this of straight.
ROLL_FOLLOW_RAD = np.radians(95)
#: ... and is pulled toward home this much less than the other joints.
ROLL_REST_SCALE = 0.1
#: Following a hand, the pull toward home, which settles the joints the target leaves free (a
#: wrist's roll) instead of letting them drift to a stop. 1 rad ~ 1 cm.
W_REST = 0.01
#: Following a hand or another arm: how hard the gripper's tilt (down onto the
#: table, or level) is followed, and its jaws' roll. Measured over every
#: local clip: at 0.05 the SO-101 tilts as the palm does (56 deg from down
#: holding, from 66) still within 2.4 mm of the path; more cost position
#: (7 mm at 0.1). The roll is from the thumb-index line, which flickers: at
#: 0.034 it cost an OpenArm 68 mm at the 95th percentile, at 0.01 21 mm.
W_FOLLOW_TILT = 0.05
W_FOLLOW_ROLL = 0.01


def reach_approach(anchor: np.ndarray, target: np.ndarray, tilt: np.ndarray | float) -> np.ndarray:
    """The way a gripper points, reaching from `anchor` (its shoulder) to
    `target`: out from the shoulder across the table, tilted down by `tilt`
    (the z of a unit direction: -1 straight down, 0 level). For one target or
    many (N, 3).

    Only the tilt is copied from a hand or another robot. Its sideways heading
    is a different arm's, or a single camera's noise, and a 5-joint arm cannot
    follow one anyway: its base sets it.
    """
    target = np.asarray(target, float)
    radial = target[..., :2] - np.asarray(anchor, float)[:2]
    radial = radial / (np.linalg.norm(radial, axis=-1, keepdims=True) + 1e-9)
    tilt = np.clip(np.asarray(tilt, float), -1.0, 1.0)
    level = np.sqrt(1.0 - tilt**2)
    return np.concatenate([radial * np.expand_dims(level, -1), np.expand_dims(tilt, -1)], axis=-1)


#: First poses this close to the best one count as reaching it as well.
FIRST_POSE_TOLERANCE_M = 0.005


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray            # 4x4, parent link -> joint frame
    axis: np.ndarray              # unit, in the joint frame
    lower: float
    upper: float
    velocity: float | None
    mimic: tuple[str, float, float] | None = None


def _origin(xyz, rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    t = np.eye(4)
    t[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    t[:3, 3] = xyz
    return t


def _floats(text: str | None, default):
    try:
        v = [float(x) for x in (text or "").split()]
        return v if len(v) == 3 else default
    except ValueError:
        return default


def _declare_prefixes(text: str) -> str:
    """Gazebo extras (<sensor:camera>, xacro:…) often use a prefix never
    declared, which XML refuses; the app accepts them, so declare them here too."""
    used = set(re.findall(r"<\/?([A-Za-z_][\w.-]*):", text)) | set(re.findall(r"\s([A-Za-z_][\w.-]*):[\w.-]+\s*=", text))
    missing = sorted(used - set(re.findall(r"xmlns:([A-Za-z_][\w.-]*)", text)) - {"xml", "xmlns"})
    if not missing:
        return text
    decl = "".join(f' xmlns:{p}="urn:undeclared:{p}"' for p in missing)
    return re.sub(r"<robot\b", "<robot" + decl, text, count=1)


def parse_urdf(text: str) -> dict[str, Joint]:
    root = ET.fromstring(_declare_prefixes(text))
    joints = {}
    for j in root.findall("joint"):
        o, a, lim, mim = j.find("origin"), j.find("axis"), j.find("limit"), j.find("mimic")
        axis = np.array(_floats(a.get("xyz") if a is not None else None, [1.0, 0.0, 0.0]))
        axis = axis / (np.linalg.norm(axis) or 1.0)

        def num(el, key, default=None):
            try:
                return float(el.get(key)) if el is not None and el.get(key) is not None else default
            except ValueError:
                return default

        kind = j.get("type", "fixed")
        joints[j.get("name")] = Joint(
            name=j.get("name"),
            type=kind,
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            origin=_origin(_floats(o.get("xyz") if o is not None else None, [0, 0, 0]),
                           _floats(o.get("rpy") if o is not None else None, [0, 0, 0])),
            axis=axis,
            lower=num(lim, "lower", -math.pi if kind == "continuous" else 0.0),
            upper=num(lim, "upper", math.pi if kind == "continuous" else 0.0),
            velocity=num(lim, "velocity"),
            mimic=((mim.get("joint"), num(mim, "multiplier", 1.0), num(mim, "offset", 0.0))
                   if mim is not None else None),
        )
    return joints


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a x b for 3-vectors, or for each column of a 3 x n `a` with one `b` —
    written out: np.cross costs more than the IK's arithmetic at this size."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _motion(joint: Joint, q: float) -> np.ndarray:
    t = np.eye(4)
    if joint.type == "prismatic":
        t[:3, 3] = joint.axis * q
        return t
    x, y, z = joint.axis
    c, s, v = math.cos(q), math.sin(q), 1 - math.cos(q)
    t[:3, :3] = [
        [v * x * x + c, v * x * y - s * z, v * x * z + s * y],
        [v * x * y + s * z, v * y * y + c, v * y * z - s * x],
        [v * x * z - s * y, v * y * z + s * x, v * z * z + c],
    ]
    return t


class Robot:
    """One arm: its chain, its limits, FK and IK — from a URDF and a profile."""

    def __init__(self, urdf_text: str, profile: dict, name: str = "robot") -> None:
        self.name = name
        self.profile = profile
        joints = parse_urdf(urdf_text)
        by_child = {j.child: j for j in joints.values()}
        path, link = [], profile["ee_link"]
        while link != profile["base_link"]:
            j = by_child.get(link)
            if j is None:
                raise ValueError(f"{profile['ee_link']} does not hang below {profile['base_link']}")
            path.append(j)
            link = j.parent
        self.path = list(reversed(path))
        self.arm = list(profile["arm_joints"])
        self.joints = joints
        missing = [n for n in self.arm if n not in joints]
        if missing:
            raise ValueError(f"Arm joints not in the URDF: {', '.join(missing)}")
        self.index = {n: i for i, n in enumerate(self.arm)}
        self.lower = np.array([joints[n].lower for n in self.arm])
        self.upper = np.array([joints[n].upper for n in self.arm])
        # The real arm's limits, when they are tighter than its URDF's (radians).
        if profile.get("limits"):
            limits = np.array(profile["limits"], float)
            if limits.shape != (len(self.arm), 2):
                raise ValueError(f"{name}: limits must be [lower, upper] for each arm joint")
            self.lower = np.maximum(self.lower, limits[:, 0])
            self.upper = np.minimum(self.upper, limits[:, 1])
        # What the IK uses: a little inside each limit, where a real joint
        # is still free to move and a solver does not park on the stop.
        margin = np.minimum(LIMIT_MARGIN_RAD, LIMIT_MARGIN_FRACTION * (self.upper - self.lower))
        self.soft_lower, self.soft_upper = self.lower + margin, self.upper - margin
        # How fast each joint may move: the profile's, where set — a URDF's is
        # often a placeholder (YAM's 1 rad/s played its datasets ~5x slower
        # than the hand) — else the URDF's, else a modest default.
        speeds = list(profile.get("velocity_limits") or []) + [None] * len(self.arm)
        self.velocity = np.array([
            float(speeds[i]) if speeds[i] else (joints[n].velocity or 3.0) for i, n in enumerate(self.arm)
        ])
        self.home = np.clip(np.array(profile.get("home_pose") or [0.0] * len(self.arm), float),
                            self.lower, self.upper)
        tcp = profile.get("tcp_offset") or [0.0] * 6
        self.tcp = _origin(tcp[:3], tcp[3:]) if any(tcp) else None
        ax = np.array(profile.get("approach_axis") or [0.0, 0.0, 1.0], float)
        self.approach_local = ax / (np.linalg.norm(ax) or 1.0)
        # The gripper's sideways (jaw) axis: with the approach axis, its roll.
        jaw = np.array(profile.get("jaw_axis") or [1.0, 0.0, 0.0], float)
        self.jaw_local = jaw / (np.linalg.norm(jaw) or 1.0)
        # The joint that rolls the gripper about the way it points (the last
        # one, its axis along the approach): two jaws are the same turned half
        # a turn, so any jaw line is met within ROLL_FOLLOW_RAD of straight,
        # and following a hand it goes no further. With the URDF's +-160 deg
        # it wandered to the other half and spun 220-240 deg in a clip.
        # Its pull toward home is lighter too: with a target of its own and
        # its stops this near, the pull only held it short of the hand's turn.
        self.rest_scale = np.ones(len(self.arm))
        pose, _, angular = self.fk_jacobian(self.home)
        if self.arm and abs(float(angular[:, -1] @ self.approach(pose))) > 0.95:
            self.soft_lower[-1] = max(self.soft_lower[-1], -ROLL_FOLLOW_RAD)
            self.soft_upper[-1] = min(self.soft_upper[-1], ROLL_FOLLOW_RAD)
            self.rest_scale[-1] = ROLL_REST_SCALE
        reach = (profile.get("workspace") or {}).get("reach_max_m")
        self.reach_max = float(reach) if reach else self._sampled_reach()
        # Where the arm works from: its shoulder (the first arm joint, at
        # zero), how far its hand gets from there, and the point it works
        # around — in front of the shoulder, a little below it, never under
        # the base. Poses move between robots relative to these.
        t = np.eye(4)
        for j in self.path:
            t = t @ j.origin
            if j.name == self.arm[0]:
                break
        self.anchor = t[:3, 3].copy()
        self.span = self._sampled_span()
        point = profile.get("work_point")
        if point:
            self.work_point = np.array(point, float)
        else:
            self.work_point = self.anchor + [0.5 * self.span, 0.0, -0.25 * self.span]
            self.work_point[2] = max(self.work_point[2], 0.05)
        gripper = profile.get("gripper") or {}
        self.gripper = gripper.get("joint")
        # The LeRobot names of the arm joints and the gripper, and the values
        # the dataset writes for an open and a closed gripper.
        self.names = list(profile.get("names") or (self.arm + ([self.gripper] if self.gripper else [])))
        if len(self.names) != len(self.arm) + (1 if self.gripper else 0):
            raise ValueError(f"{name}: names must be one per arm joint, then the gripper")
        lerobot = gripper.get("lerobot") or {}
        self.gripper_open = float(lerobot.get("open", 100.0))
        self.gripper_closed = float(lerobot.get("closed", 0.0))
        self.side_prefix = profile.get("side_prefix") or "{side}_arm_"
        self.right = None
        if profile.get("right_arm"):
            right = {k: v for k, v in profile.items() if k not in ("right_arm", "work_point", "workspace")}
            right.update(profile["right_arm"])
            if "names" not in profile and "names" not in profile["right_arm"]:
                right.pop("names", None)
            self.right = Robot(urdf_text, right, f"{name} (right)")

    @property
    def on_table(self) -> bool:
        """It stands on the table: z = 0 in its base frame is the table's surface."""
        return bool(self.anchor[2] < TABLE_MOUNTED_SHOULDER_M)

    @property
    def dof(self) -> int:
        return len(self.arm)

    def fk(self, q: np.ndarray) -> np.ndarray:
        """The end-effector pose (4x4) in the base frame for arm joints `q`."""
        values = {n: float(q[i]) for i, n in enumerate(self.arm)}
        t = np.eye(4)
        for j in self.path:
            t = t @ j.origin
            if j.type in ACTUATED:
                if j.mimic:
                    v = values.get(j.mimic[0], 0.0) * j.mimic[1] + j.mimic[2]
                else:
                    v = values.get(j.name, 0.0)
                t = t @ _motion(j, v)
        return t @ self.tcp if self.tcp is not None else t

    def fk_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The end-effector pose, and how it moves per arm joint: the linear
        (3 x n) and angular (3 x n) velocity of the tip for each joint — from
        each joint's axis and position along the chain, in one pass (a mimic
        joint moves with the joint it follows)."""
        values = {n: float(q[i]) for i, n in enumerate(self.arm)}
        t = np.eye(4)
        moved: list[tuple[int, float, str, np.ndarray, np.ndarray]] = []
        for j in self.path:
            t = t @ j.origin
            if j.type in ACTUATED:
                source, gain = (j.mimic[0], j.mimic[1]) if j.mimic else (j.name, 1.0)
                if source in self.index:
                    moved.append((self.index[source], gain, j.type, t[:3, :3] @ j.axis, t[:3, 3].copy()))
                v = values.get(source, 0.0) * gain + (j.mimic[2] if j.mimic else 0.0)
                t = t @ _motion(j, v)
        pose = t @ self.tcp if self.tcp is not None else t
        linear, angular = np.zeros((3, self.dof)), np.zeros((3, self.dof))
        for i, gain, kind, axis, point in moved:
            if kind == "prismatic":
                linear[:, i] += gain * axis
            else:
                linear[:, i] += gain * _cross(axis, pose[:3, 3] - point)
                angular[:, i] += gain * axis
        return pose, linear, angular

    def approach(self, pose: np.ndarray) -> np.ndarray:
        return pose[:3, :3] @ self.approach_local

    def jaw(self, pose: np.ndarray) -> np.ndarray:
        return pose[:3, :3] @ self.jaw_local

    def side(self, side: str) -> Robot:
        """The arm on that side: a two-arm URDF has its own right arm."""
        return self.right if side == "right" and self.right is not None else self

    def prefix(self, side: str) -> str:
        return self.side_prefix.format(side=side)

    def gripper_pct(self, value: float) -> float:
        """A gripper value as written in a dataset, as 0 (closed) .. 100 (open)."""
        span = self.gripper_open - self.gripper_closed
        return float((value - self.gripper_closed) / span * 100.0) if span else 100.0

    def gripper_value(self, pct: float) -> float:
        return self.gripper_closed + float(np.clip(pct, 0.0, 100.0)) / 100.0 * (self.gripper_open - self.gripper_closed)

    def _sampled_span(self, n: int = 1500) -> float:
        rng = np.random.default_rng(7)
        pts = np.array([self.fk(rng.uniform(self.lower, self.upper))[:3, 3] for _ in range(n)])
        return float(np.percentile(np.linalg.norm(pts - self.anchor, axis=1), 95))

    def _sampled_reach(self, n: int = 1500) -> float:
        rng = np.random.default_rng(7)
        pts = [self.fk(rng.uniform(self.lower, self.upper))[:3, 3] for _ in range(n)]
        return float(np.percentile(np.linalg.norm(np.array(pts)[:, :2], axis=1), 95))

    def ik(self, target: np.ndarray, approach: np.ndarray | None, seed: np.ndarray,
           jaw: np.ndarray | None = None, w_pos: float = 1.0, w_axis: float = 0.05, w_jaw: float = 0.03,
           w_home: float = 0.002, w_still: float = 0.01, soft: bool = False) -> np.ndarray:
        """Arm joints that put the end-effector at `target` (metres, base
        frame), pointing along `approach` and rolled to `jaw` as far as it
        can, close to `seed` — kept off the stops when `soft` (a hand's path,
        which only says where the gripper goes; a robot's own poses are kept
        exactly, stops and all)."""
        # The solver asks for the residual and then the slope at the same q:
        # one pass along the chain answers both.
        last: dict = {}
        # Following a hand the roll joint is pulled home less (rest_scale).
        rest = w_home * (self.rest_scale if soft else 1.0) * np.ones(self.dof)

        def at(q):
            if last.get("q") is None or not np.array_equal(last["q"], q):
                last["q"], last["fk"] = q.copy(), self.fk_jacobian(q)
            return last["fk"]

        def residual(q):
            pose = at(q)[0]
            r = [w_pos * (pose[:3, 3] - target)]
            if approach is not None and w_axis:
                r.append(w_axis * (self.approach(pose) - approach))
            if jaw is not None and w_jaw:
                # A gripper's jaws are a line, not an arrow: either way along it
                # is the same grip (robots draw that axis opposite ways).
                r.append(w_jaw * _cross(self.jaw(pose), jaw))
            r.append(rest * (q - self.home))
            r.append(w_still * (q - seed))
            return np.concatenate(r)

        eye = np.eye(self.dof)

        def jacobian(q):
            # Exact, from the joint axes: one pass along the chain, where
            # estimating it would take one per joint.
            pose, linear, angular = at(q)
            rows = [w_pos * linear]
            if approach is not None and w_axis:
                rows.append(w_axis * _cross(angular, self.approach(pose)))
            if jaw is not None and w_jaw:
                rows.append(w_jaw * _cross(_cross(angular, self.jaw(pose)), jaw))
            rows.append(np.diag(rest))
            rows.append(w_still * eye)
            return np.vstack(rows)

        lo, hi = (self.soft_lower, self.soft_upper) if soft else (self.lower, self.upper)
        x0 = np.clip(seed, lo + 1e-9, hi - 1e-9)
        return least_squares(residual, x0, jac=jacobian, bounds=(lo, hi), method="trf", max_nfev=60, xtol=1e-6).x

    def first_ik(self, target: np.ndarray, approach: np.ndarray | None,
                 jaw: np.ndarray | None = None, **ik) -> np.ndarray:
        """The first frame of an episode: from home and a few other starts,
        the pose nearest home of those that reach the target about as well as
        the best. Every later frame is seeded from this one, so a twisted
        start that is a fraction of a millimetre closer would twist the whole
        episode (an elbow bent backwards, a wrist on its stop)."""
        rng = np.random.default_rng(1)
        seeds = [self.home] + [rng.uniform(self.lower, self.upper) for _ in range(5)]
        found = []
        for s in seeds:
            q = self.ik(target, approach, s, jaw, **{**ik, "w_still": 0.0})
            found.append((float(np.linalg.norm(self.fk(q)[:3, 3] - target)), q))
        best = min(e for e, _ in found)
        good = [q for e, q in found if e <= best + FIRST_POSE_TOLERANCE_M]
        return min(good, key=lambda q: float(np.linalg.norm(q - self.home)))


def load(urdf_path: str | Path, profile: dict, name: str = "robot") -> Robot:
    return Robot(Path(urdf_path).read_text(encoding="utf-8"), profile, name)


# The SO-101 the pipeline ships with, for robots made before URDF import.
SO101_PROFILE = {
    "base_link": "base_link",
    "ee_link": "gripper_frame_link",
    "arm_joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    "gripper": {"joint": "gripper", "open": 1.74533, "closed": -0.174533},
    "home_pose": [0.0, 0.0, 0.0, 0.0, 0.0],
    "approach_axis": [0.0, 0.0, 1.0],
    "jaw_axis": [1.0, 0.0, 0.0],
    "tcp_offset": [0.0] * 6,
    "units": "degrees",
    "robot_type": "so101_follower",
}
