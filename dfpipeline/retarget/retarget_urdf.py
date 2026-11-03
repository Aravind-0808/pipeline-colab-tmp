"""A hand's path onto any arm, from its URDF.

The SO-101 has its own retargeter (retarget_so101.py), tuned to that arm.
Every other robot is an imported URDF with a profile (base, end-effector, arm
joints, gripper, LeRobot names): this solves its arm the same way, through the
same steps, with the robot's own kinematics (dfpipeline/embodiment/robot.py).

    hand path (camera)  ->  the table's frame (up = away from the table, x =
    the way the hand reaches)  ->  the pinch point, centred on the recording
    and scaled to this arm's size, around the arm's working point  ->  IK each
    video frame, seeded from the last  ->  home moves, one retiming on the
    robot's joint speeds, samples at the video rate

Two hands go onto the arm's two sides: the right arm of a two-arm URDF
(OpenArm), or the same arm twice (YAM, a single-arm URDF) — each solved from
its own hand and retimed on one clock, so the handover lines up.

The trajectory is written as the SO-101's is (so the gate, the review screens
and the dataset build read it the same way), with the robot's own LeRobot
names in `action`: `joint1.pos` …, or `left_joint_1.pos` … for two arms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dfpipeline.embodiment.convert import Encoding, encoding
from dfpipeline.embodiment.robot import W_FOLLOW_ROLL, W_FOLLOW_TILT, W_REST, Robot, reach_approach
from dfpipeline.retarget.retarget_so101 import (
    CONTACT_HEIGHT_M,
    HOME_HOLD_S,
    HOME_MOVE_DEG_S,
    SO101Retargeter,
    retime,
)

log = logging.getLogger("retarget.urdf")

#: What a person's working reach spans from the shoulder: the hand path is
#: scaled by the arm's own span against this, so a small arm gets a small
#: version of the motion and a big one a big version.
HUMAN_SPAN_M = 0.6
#: Targets are kept this far inside the arm's span: the edge of the reach is
#: where IK flips and joints sit on their limits.
REACH_MARGIN = 0.95
#: How far apart a pair of one-arm robots stands, base to base, side by side.
PAIR_SPACING_M = 0.4
#: Targets stay within this angle either side of the arm's forward direction.
#: A hand coming in or going out at the frame's edge is beside the camera,
#: not beside the robot: followed, the base swung 100 deg and back.
MAX_AZIMUTH_RAD = np.radians(60)
#: The gripper point never goes lower than this above the table.
MIN_EE_HEIGHT_M = 0.01
#: The highest a hand lifts a thing above the table while working on it.
MAX_LIFT_M = 0.30
#: Height (the noisiest coordinate) is smoothed over this long.
HEIGHT_SMOOTH_S = 0.5
#: The gripper's tilt and roll are followed as robot.py says (W_FOLLOW_*).
W_APPROACH = W_FOLLOW_TILT
W_JAW = W_FOLLOW_ROLL
W_STILL = 0.02
#: The palm's direction is steadied over this long.
APPROACH_SMOOTH_S = 0.3
#: Joint acceleration limit, as a multiple of the joint's speed limit per second.
ACCEL_PER_SPEED = 4.0


@dataclass
class UrdfPath:
    """One arm's path, a sample per video frame, before home moves and retiming."""

    dense_idx: np.ndarray     # (N,) source video frame of each sample
    q_path: np.ndarray        # (N, dof) radians
    tracked: np.ndarray       # (N,) the hand was seen (not interpolated) here
    d_open: np.ndarray        # (N,) gripper opening 0..1
    d_target: np.ndarray      # (N, 3) where the gripper was asked to be, base frame
    reachable_fraction: float
    fps: float

    def spanning(self, first: int, last: int) -> UrdfPath:
        """Over video frames first..last: an arm whose hand comes in later
        waits at its first pose, one whose hand leaves earlier at its last."""
        before = max(0, int(self.dense_idx[0]) - first)
        after = max(0, last - int(self.dense_idx[-1]))

        def hold(arr, fill=None):
            arr = np.asarray(arr)
            head = np.repeat(arr[:1], before, 0) if fill is None else np.full((before,) + arr.shape[1:], fill)
            tail = np.repeat(arr[-1:], after, 0) if fill is None else np.full((after,) + arr.shape[1:], fill)
            return np.concatenate([head, arr, tail])

        return UrdfPath(np.arange(first, last + 1), hold(self.q_path), hold(self.tracked, False),
                        hold(self.d_open), hold(self.d_target), self.reachable_fraction, self.fps)


def _fps(frames: list) -> float:
    return SO101Retargeter._fps(frames)


def _height(rel_z: np.ndarray, table_distance_m: float | None, fps: float) -> np.ndarray:
    """The pinch point's height above the table, as far as one camera can tell.

    Depth from a single camera is the least certain coordinate by far: on
    real clips the hand is placed from 30 cm nearer than the table to 30 cm
    beyond it, while it really moves 10-20 cm. Read against the table's own
    distance when known, never under the table, never more than a hand lifts
    a thing, and smoothed over half a second so the noise does not become
    motion. Sideways motion is kept as recorded."""
    from scipy.signal import medfilt, savgol_filter

    if table_distance_m:
        height = table_distance_m + rel_z      # rel_z is up from the camera: -depth along the normal
    else:
        height = rel_z - np.percentile(rel_z, 5)
    height = np.clip(height, 0.0, MAX_LIFT_M) + CONTACT_HEIGHT_M
    window = int(HEIGHT_SMOOTH_S * fps) | 1
    if len(height) > window >= 5:
        height = savgol_filter(medfilt(height, 5), window, 2)
    return np.clip(height, CONTACT_HEIGHT_M, MAX_LIFT_M + CONTACT_HEIGHT_M)


def _palm_approach(J: np.ndarray, fps: float) -> np.ndarray:
    """Which way the gripper points: out of the palm, toward what it takes.

    Not wrist-to-fingertips. A hand taking a cup from above keeps its fingers
    level and turns its palm down onto it: the fingers pointed 85-87 deg from
    straight down while holding, so the gripper came in level; the palm points
    40-50 deg from down, as a person driving an SO-101 brings its jaws down.
    The palm's side is the one the fingers curl to; it is steadied over a
    short window, as hand tracking flickers from frame to frame.
    """
    from scipy.signal import savgol_filter

    n = np.cross(J[:, 5] - J[:, 0], J[:, 17] - J[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    palm = J[:, [0, 5, 9, 13, 17]].mean(axis=1)
    curl = J[:, [8, 12, 16]].mean(axis=1) - palm
    n *= np.sign(np.median(np.einsum("ij,ij->i", n, curl))) or 1.0
    window = int(APPROACH_SMOOTH_S * fps) | 1
    if len(n) > window >= 5:
        n = savgol_filter(n, window, 2, axis=0)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def _jaw_line(J: np.ndarray) -> np.ndarray:
    """The line the gripper's jaws close along, from a hand: across its
    knuckles (index to little finger), in the palm's plane.

    It was thumb tip to index tip. In a grasp those are behind the object and
    wander: followed, the SO-101's wrist swept 165-240 deg a clip and still
    missed the line by 40-54 deg, while the operator's wrist turned far less.
    The knuckle line is the palm's own, and turns exactly as the wrist turns:
    over every local clip the wrist rolled 78 deg a clip and met the line
    within 28 deg, position unchanged. (Set at the tips' mean angle to the
    knuckles instead, it asked for a roll the arm cannot make while reaching:
    65 deg off.)
    """
    n = np.cross(J[:, 5] - J[:, 0], J[:, 17] - J[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    v = J[:, 5] - J[:, 17]
    v = v - np.einsum("ij,ij->i", v, n)[:, None] * n
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


@dataclass
class Placement:
    """Where a hand's path goes on its arm, when two hands share one frame.

    `R_rw` is one robot frame for both hands, `centre` the point between the
    two hands (in it) that lands at `lands` — this arm's base frame — and
    `height_mid` the middle height of both, for arms hanging from a torso.
    """

    R_rw: np.ndarray
    centre: np.ndarray
    lands: np.ndarray
    height_mid: float


def solve(arm: Robot, frames: list, table_normal_cam: np.ndarray | None, grip: dict | None,
          table_distance_m: float | None = None, placement: Placement | None = None) -> UrdfPath:
    """This arm's joints for one hand's path, one sample per video frame —
    alone (centred on itself, at the arm's working point), or placed with the
    other hand (`placement`)."""
    from scipy.signal import medfilt, savgol_filter

    from dfpipeline.config import settings
    from dfpipeline.retarget import grasp

    J = np.stack([f.joints_world for f in frames]).astype(np.float64)       # (T, 21, 3)
    R_rw = placement.R_rw if placement else SO101Retargeter._robot_frame(J, table_normal_cam)
    pinch = 0.5 * (J[:, 4] + J[:, 8])
    wrist = J[:, 0]

    # The pinch point in the table's frame, centred on where this recording
    # mostly happens, scaled to the arm's size and put at its working point.
    rel = pinch @ R_rw.T
    fps = _fps(frames)
    height = _height(rel[:, 2], table_distance_m, fps)
    scale = float(np.clip(arm.span / HUMAN_SPAN_M, 0.4, 2.0))
    # Where the table is for this arm: under its base when the arm stands on
    # it (its shoulder near the base — SO-101, YAM, Piper), so the hand keeps
    # its real height above the table and never goes into it; else (an arm
    # hanging from a torso — OpenArm) the motion's middle is at the working point.
    height_mid = placement.height_mid if placement else np.median(height)
    table_z = 0.0 if arm.on_table else arm.work_point[2] - scale * height_mid
    if placement:
        xy = placement.lands + scale * (rel[:, :2] - placement.centre)
    else:
        xy = arm.work_point[:2] + scale * (rel[:, :2] - np.median(rel[:, :2], axis=0))
    faithful = np.column_stack([xy, table_z + np.maximum(scale * height, MIN_EE_HEIGHT_M)])
    # In front of the arm: the angle around its shoulder from the working
    # point is limited, the distance kept.
    forward = np.arctan2(*(arm.work_point - arm.anchor)[1::-1])
    around = faithful[:, :2] - arm.anchor[:2]
    angle = np.arctan2(around[:, 1], around[:, 0]) - forward
    angle = np.clip((angle + np.pi) % (2 * np.pi) - np.pi, -MAX_AZIMUTH_RAD, MAX_AZIMUTH_RAD) + forward
    radius = np.linalg.norm(around, axis=1)
    faithful[:, :2] = arm.anchor[:2] + radius[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
    # Inside the arm's reach, measured from its shoulder.
    offset = faithful - arm.anchor
    dist = np.linalg.norm(offset, axis=1, keepdims=True)
    limit = REACH_MARGIN * arm.span
    targets = arm.anchor + offset * np.minimum(1.0, limit / np.maximum(dist, 1e-9))
    reachable = float(np.mean(np.linalg.norm(targets - faithful, axis=1) < 0.005))

    # The palm's tilt — down onto the table or level across it — is what is
    # copied; its sideways heading from one camera is noise, and a 5-joint
    # arm cannot follow one anyway: its base sets that. So the gripper points
    # out from the shoulder toward the target, tilted as the palm is. Copying
    # the whole direction cost the SO-101 8.7 mm of position at weight 0.05.
    approach = reach_approach(arm.anchor, targets, _palm_approach(J @ R_rw.T, fps)[:, 2])
    jaw = SO101Retargeter._steady_jaw(_jaw_line(J @ R_rw.T), fps)

    # The gripper opens and closes as the hand does, shut while it holds (grasp.py).
    aperture = np.linalg.norm(J[:, 8] - J[:, 4], axis=1)
    opening = grasp.opening(np.array([f.frame_idx for f in frames]), aperture, grip)
    if len(opening) >= 9:
        opening = np.clip(savgol_filter(medfilt(opening, 5), 9, 2), 0.0, 1.0)

    # Every video frame from the hand's first to its last; gaps are bridged.
    src_idx = np.array([f.frame_idx for f in frames])
    dense_idx = np.arange(src_idx[0], src_idx[-1] + 1)
    tracked = np.isin(dense_idx, src_idx)

    def interp(arr):
        arr = np.asarray(arr, float)
        flat = arr.reshape(len(arr), -1)
        out = np.stack([np.interp(dense_idx, src_idx, flat[:, k]) for k in range(flat.shape[1])], 1)
        return out.reshape((len(dense_idx),) + arr.shape[1:])

    d_targets, d_open = interp(targets), np.clip(interp(opening), 0.0, 1.0)
    d_approach, d_jaw = interp(approach), interp(jaw)
    d_approach /= np.linalg.norm(d_approach, axis=1, keepdims=True) + 1e-9
    d_jaw /= np.linalg.norm(d_jaw, axis=1, keepdims=True) + 1e-9
    if len(d_targets) >= 9:
        d_targets = savgol_filter(d_targets, 9, 2, axis=0)   # target noise is amplified by IK

    # IK each frame, seeded from the one before: no branch jumps.
    q_path = np.zeros((len(dense_idx), arm.dof))
    # Off the stops and settled toward home where the hand leaves a joint free.
    q = arm.first_ik(d_targets[0], d_approach[0], d_jaw[0], w_home=W_REST, soft=True)
    for i in range(len(dense_idx)):
        q = arm.ik(d_targets[i], d_approach[i], q, d_jaw[i], w_axis=W_APPROACH, w_jaw=W_JAW, w_still=W_STILL,
                   w_home=W_REST, soft=True)
        q_path[i] = q
    if len(q_path) >= 15:
        # Smoothing must not undo the IK's margin from the stops.
        q_path = np.clip(savgol_filter(q_path, 15, 3, axis=0), arm.soft_lower, arm.soft_upper)

    return UrdfPath(dense_idx, q_path, tracked, d_open, d_targets, reachable, fps)


@dataclass
class Sample:
    frame: int
    timestamp: float
    phase: str
    q: np.ndarray
    open: float
    target: np.ndarray
    ee: np.ndarray
    tracked: bool


def finish(arms: list[tuple[Robot, UrdfPath]]) -> tuple[list[list[Sample]], dict]:
    """Home moves, one retiming on every arm's joint speeds, and fixed-rate
    samples — for one arm or two, on one clock."""
    fps = arms[0][1].fps

    def ease(a, b):
        n = max(2, int(np.ceil(np.max(np.abs(b - a)) / np.radians(HOME_MOVE_DEG_S) * fps)) + 1)
        s = (1 - np.cos(np.linspace(0.0, np.pi, n))) / 2
        return a + s[:, None] * (b - a)

    pres, posts = [], []
    for arm, path in arms:
        hold = np.repeat(arm.home[None], max(1, int(HOME_HOLD_S * fps)), axis=0)
        pres.append(np.vstack([hold, ease(arm.home, path.q_path[0])[1:-1]]))
        posts.append(np.vstack([ease(path.q_path[-1], arm.home)[1:-1], hold]))
    n_pre, n_post = max(len(x) for x in pres), max(len(x) for x in posts)
    n_action = len(arms[0][1].dense_idx)

    def pad(arr):
        arr = np.asarray(arr)
        return np.concatenate([np.repeat(arr[:1], n_pre, 0), arr, np.repeat(arr[-1:], n_post, 0)])

    dense_idx = pad(arms[0][1].dense_idx)
    phase = np.array(["home_start"] * n_pre + ["action"] * n_action + ["home_end"] * n_post)
    per_arm = []
    for (arm, path), pre, post in zip(arms, pres, posts, strict=True):
        pre = np.vstack([np.repeat(arm.home[None], n_pre - len(pre), 0), pre])
        post = np.vstack([post, np.repeat(arm.home[None], n_post - len(post), 0)])
        q_path = np.vstack([pre, path.q_path, post])
        tracked = np.concatenate([np.zeros(n_pre, bool), path.tracked, np.zeros(n_post, bool)])
        d_open = np.concatenate([np.full(n_pre, 1.0), path.d_open, np.full(n_post, 1.0)])
        d_target = pad(path.d_target)
        for j in list(range(n_pre)) + list(range(n_pre + n_action, len(q_path))):
            d_target[j] = arm.fk(q_path[j])[:3, 3]   # home moves: no hand to follow
        per_arm.append((q_path, tracked, d_open, d_target))

    q_all = np.hstack([a[0] for a in per_arm])
    vmax = np.concatenate([arm.velocity for arm, _ in arms])
    amax = ACCEL_PER_SPEED * vmax
    robot_t, t_out, q_out, excess = retime(q_all, fps, vmax, amax)

    video_t = (dense_idx - dense_idx[0]) / fps
    a0, a1 = n_pre, n_pre + n_action - 1
    action_robot_s = float(robot_t[a1] - robot_t[a0])
    action_video_s = float(video_t[a1] - video_t[a0])
    stats = {
        "video_duration_s": round(action_video_s, 3),
        "robot_duration_s": round(action_robot_s, 3),
        "slowdown": round(action_robot_s / max(action_video_s, 1e-6), 3) if action_video_s > 0 else 1.0,
        "peak_speed_or_accel_vs_limit": round(excess, 3),
        "hand_path_reachable_fraction": round(min(p.reachable_fraction for _, p in arms), 3),
    }

    src_out = np.interp(t_out, robot_t, dense_idx.astype(float))
    knot = np.clip(np.searchsorted(robot_t, t_out, side="right") - 1, 0, len(phase) - 1)
    results, col = [], 0
    for (arm, _), (_q, tracked, d_open, d_target) in zip(arms, per_arm, strict=True):
        q_arm = q_out[:, col:col + arm.dof]
        col += arm.dof
        open_out = np.interp(t_out, robot_t, d_open)
        tgt_out = np.stack([np.interp(t_out, robot_t, d_target[:, c]) for c in range(3)], 1)
        tracked_out = np.interp(t_out, robot_t, tracked.astype(float)) > 0.5
        results.append([
            Sample(int(np.rint(src_out[i])), float(t_out[i]), str(phase[knot[i]]), q_arm[i].copy(),
                   float(open_out[i]), tgt_out[i], arm.fk(q_arm[i])[:3, 3], bool(tracked_out[i]))
            for i in range(len(t_out))
        ])
    return results, stats


def _record(arm: Robot, enc: Encoding, prefix: str, s: Sample) -> tuple[dict, dict]:
    """The action columns this arm writes, and its own sample for the review screens."""
    values = enc.from_rad(s.q)
    action = {f"{prefix}{name}.pos": round(float(v), 3) for name, v in zip(arm.names[:arm.dof], values, strict=True)}
    gripper_q = None
    if arm.gripper:
        g = (arm.profile.get("gripper") or {})
        closed, opened = float(g.get("closed", 0.0)), float(g.get("open", 1.0))
        gripper_q = closed + s.open * (opened - closed)
        action[f"{prefix}{arm.names[-1]}.pos"] = round(arm.gripper_value(100.0 * s.open), 3)
    own = {
        "joints_rad": [round(float(v), 5) for v in s.q] + ([round(gripper_q, 5)] if gripper_q is not None else []),
        "target_pos": [round(float(v), 4) for v in s.target],
        "ee_pos": [round(float(v), 4) for v in s.ee],
        "position_error_m": round(float(np.linalg.norm(s.ee - s.target)), 4),
        "gripper_open": round(s.open, 3),
        "tracked": s.tracked,
    }
    return action, own


def pair_bases(robot: Robot) -> dict[str, np.ndarray]:
    """Where each side's arm base stands, in one frame (x forward, y left).

    A two-arm URDF's arms already share one base frame: both at its origin.
    A pair of one-arm robots stands side by side, facing forward,
    `pair_spacing_m` apart (the profile's, else PAIR_SPACING_M).
    """
    if robot.right is not None:
        return {"left": np.zeros(2), "right": np.zeros(2)}
    gap = float(robot.profile.get("pair_spacing_m") or PAIR_SPACING_M)
    return {"left": np.array([0.0, gap / 2]), "right": np.array([0.0, -gap / 2])}


def placements(robot: Robot, hands: dict[str, list], table_normal_cam: np.ndarray | None,
               table_distance_m: float | None) -> dict[str, Placement]:
    """Two hands in one frame, so a handover is one.

    Each hand had its own frame and was centred on its own middle, at its
    own arm's working point: when the hands met over a cup, each gripper
    stayed in front of its own arm. Now one frame is drawn from both hands,
    and the point between them lands between the two arms' working points —
    the hands keep their distance and direction from each other, scaled.
    """
    sides = ("left", "right")
    J = {s: np.stack([f.joints_world for f in hands[s]]).astype(np.float64) for s in sides}
    R_rw = SO101Retargeter._robot_frame(np.concatenate([J[s] for s in sides]), table_normal_cam)
    rel = {s: (0.5 * (J[s][:, 4] + J[s][:, 8])) @ R_rw.T for s in sides}
    centre = 0.5 * (np.median(rel["left"][:, :2], axis=0) + np.median(rel["right"][:, :2], axis=0))
    fps = _fps(hands["left"])
    height_mid = float(np.median(np.concatenate([_height(rel[s][:, 2], table_distance_m, fps) for s in sides])))
    bases = pair_bases(robot)
    # The middle of the two working points, in the shared frame.
    middle = 0.5 * sum(robot.side(s).work_point[:2] + bases[s] for s in sides)
    return {s: Placement(R_rw, centre, middle - bases[s], height_mid) for s in sides}


def retarget_to_urdf(robot: Robot, hands: dict[str, list], table_normal_cam: np.ndarray | None,
                     grips: dict[str, dict | None], out_dir: Path, name: str,
                     table_distance_m: float | None = None) -> Path:
    """One hand (key "") onto the arm, or two ("left", "right") onto its two
    sides; writes robot_trajectory.json and returns its path."""
    sides = [s for s in ("left", "right") if hands.get(s)] or [""]
    if len(sides) == 1 and sides[0] and not hands.get(""):
        raise ValueError(f"both hands are needed for two arms, got {sides}")
    units = robot.profile.get("units") or "degrees"
    arms = {s: robot.side(s or "left") for s in sides}
    places = placements(robot, hands, table_normal_cam, table_distance_m) if sides == ["left", "right"] else {}
    solved = {s: solve(arms[s], hands[s], table_normal_cam, grips.get(s), table_distance_m, places.get(s))
              for s in sides}
    first = min(int(p.dense_idx[0]) for p in solved.values())
    last = max(int(p.dense_idx[-1]) for p in solved.values())
    samples, stats = finish([(arms[s], solved[s].spanning(first, last)) for s in sides])

    encs = {s: encoding(arms[s], units) for s in sides}
    records = []
    for at in zip(*samples, strict=True):
        action, own = {}, {}
        for s, sample in zip(sides, at, strict=True):
            prefix = arms[s].prefix(s) if s else ""
            a, o = _record(arms[s], encs[s], prefix, sample)
            action |= a
            own[s or "arm"] = o
        records.append({
            "frame": at[0].frame,
            "timestamp": round(at[0].timestamp, 4),
            "phase": at[0].phase,
            "action": action,
            "position_error_m": max(o["position_error_m"] for o in own.values()),
            "tracked": all(o["tracked"] for o in own.values()),
            **own,
        })
    errors = [r["position_error_m"] for r in records if r["phase"] == "action"]
    log.info("%s retargeting: %d samples | error median %.1f mm | x%.2f slower",
             name, len(records), 1000 * float(np.median(errors)) if errors else 0.0, stats["slowdown"])
    payload = {
        "robot": name,
        **({"arms": sides} if sides != [""] else {}),
        "action_names": list(records[0]["action"]) if records else [],
        "units": {"action": f"{units} (arm), the gripper in the robot's own dataset values",
                  "positions": "meters, robot base frame (+Z up)",
                  "timestamp": "seconds on the robot clock (retimed); frame = source video frame"},
        "fps": round(float(solved[sides[0]].fps), 4),
        "gripper": {"source": "hand+holds" if any(g and g.get("holds") for g in grips.values()) else "hand"},
        "retiming": stats,
        "trajectory": records,
    }
    path = Path(out_dir) / "robot_trajectory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
