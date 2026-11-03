"""A clip's hand path onto an imported robot's arm, from its URDF — no SO-101 in between."""

import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from test_embodiment import (
    ARM6,
    ARM6_PROFILE,
    BI_ARM6,
    BI_ARM6_PROFILE,
    _bi_arm_profile,
)
from test_two_hands import _hand

from dfpipeline.embodiment.robot import Robot
from dfpipeline.retarget import service as rs
from dfpipeline.retarget.retarget_urdf import retarget_to_urdf

TABLE = np.array([0.0, 0.0, 1.0])


def _check(path, robot_side: Robot, prefix: str, side_key: str, data: dict) -> None:
    action_rows = [r for r in data["trajectory"] if r["phase"] == "action"]
    assert action_rows
    errors = [r[side_key]["position_error_m"] for r in action_rows]
    assert np.median(errors) < 0.01, "the arm follows the hand's path"
    lo, hi = np.degrees(robot_side.lower), np.degrees(robot_side.upper)
    for r in data["trajectory"]:
        joints = [r["action"][f"{prefix}{n}.pos"] for n in robot_side.names[:robot_side.dof]]
        assert all(lo - 0.01 <= joints) and all(joints <= hi + 0.01)


def test_one_hand_onto_an_imported_arm(tmp_path):
    arm = Robot(ARM6, {**ARM6_PROFILE, "names": [f"joint_{i}" for i in range(1, 7)] + ["gripper"],
                       "gripper": {**ARM6_PROFILE["gripper"], "lerobot": {"open": 100, "closed": 0}}}, "Arm6")
    path = retarget_to_urdf(arm, {"": _hand(0.0, range(90))}, TABLE, {"": None}, tmp_path, "Arm6")
    data = json.loads(path.read_text())
    assert path.name == "robot_trajectory.json" and "arms" not in data
    assert data["action_names"] == [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
    _check(path, arm, "", "arm", data)
    grip = [r["action"]["gripper.pos"] for r in data["trajectory"]]
    assert min(grip) >= 0 and max(grip) <= 100
    # The gate reads it as it reads the SO-101's.
    metrics = rs.read_metrics(tmp_path)
    assert metrics["slowdown"] >= 1.0 and metrics["position_error_median_m"] < 0.01
    assert metrics["reachable_fraction"] is not None


def test_two_hands_onto_the_two_sides_of_a_two_arm_urdf(tmp_path):
    """The right hand drives the URDF's own right arm (a mirror of the left),
    both on one clock, under the robot's LeRobot names."""
    robot = Robot(BI_ARM6, BI_ARM6_PROFILE, "BiArm6")
    hands = {"left": _hand(-0.1, range(90)), "right": _hand(0.1, range(30, 120))}
    path = retarget_to_urdf(robot, hands, TABLE, {"left": None, "right": None}, tmp_path, "BiArm6")
    data = json.loads(path.read_text())
    assert data["arms"] == ["left", "right"]
    names = BI_ARM6_PROFILE["names"]
    assert data["action_names"] == [f"{s}_{n}.pos" for s in ("left", "right") for n in names]
    for side in ("left", "right"):
        _check(path, robot.side(side), f"{side}_", side, data)
    # The right arm is the URDF's own (its shoulder turns the other way).
    right = np.array([r["action"]["right_joint_2.pos"] for r in data["trajectory"]])
    lo, hi = np.degrees(np.array(_bi_arm_profile("right")["limits"][1]))
    assert (right >= lo - 0.01).all() and (right <= hi + 0.01).all()
    # One clock: every sample is the same instant on both arms.
    assert all("left" in r and "right" in r for r in data["trajectory"])


def test_a_hand_from_the_frame_edge_keeps_the_arm_in_front_and_off_its_stops(tmp_path):
    """The SO-101 from its URDF alone. A hand that comes in from the side of
    the picture is beside the camera, not the robot: the base swung 100 deg
    to it and back, a random first pose bent the elbow backwards for a whole
    episode, and the wrist sat on its stops. Now the arm stays in front and
    off its stops, and still follows the hand."""
    from test_embodiment import SO101_URDF

    from dfpipeline.embodiment.robot import LIMIT_MARGIN_RAD, SO101_PROFILE, load
    from dfpipeline.retarget.retarget_urdf import MAX_AZIMUTH_RAD
    from test_two_hands import _Frame

    arm = load(SO101_URDF, SO101_PROFILE, "SO-101")
    hand = _hand(0.0, range(90))
    # The first second: coming in from far off to one side.
    for k, f in enumerate(hand[:30]):
        hand[k] = _Frame(f.frame_idx, f.joints_world + [0.6 * (1 - k / 30), 0.0, 0.0])
    path = retarget_to_urdf(arm, {"": hand}, TABLE, {"": None}, tmp_path, "SO-101")
    data = json.loads(path.read_text())
    action = [r for r in data["trajectory"] if r["phase"] == "action"]
    q = np.array([r["arm"]["joints_rad"][:arm.dof] for r in action])

    forward = np.arctan2(*(arm.work_point - arm.anchor)[1::-1])
    assert np.abs(q[:, 0] - forward).max() <= MAX_AZIMUTH_RAD + np.radians(5)
    assert (q >= arm.lower + LIMIT_MARGIN_RAD - 1e-3).all() and (q <= arm.upper - LIMIT_MARGIN_RAD + 1e-3).all()
    assert np.median([r["position_error_m"] for r in action[30:]]) < 0.01


def _palm_down_hand(frames: range) -> list:
    """A hand over the table, palm facing down, fingers curled toward it:
    taking something from above. Its fingers point level, its palm down."""
    from test_two_hands import _Frame

    out = []
    for i in frames:
        p = np.array([0.03 * np.sin(i / 20), 0.02 * i / 90, 0.45])
        j = np.tile(p + [0, 0, -0.05], (21, 1))
        j[0] = p + [0, -0.09, -0.05]                 # wrist, behind
        j[5] = p + [-0.03, -0.01, -0.05]             # index knuckle
        j[9] = p + [0.0, -0.01, -0.05]
        j[13] = p + [0.02, -0.01, -0.05]
        j[17] = p + [0.04, -0.02, -0.05]             # little-finger knuckle
        j[12], j[16] = p + [0.0, 0.02, 0.0], p + [0.02, 0.02, 0.0]   # curled down
        j[4], j[8] = p + [-0.015, 0.01, 0.0], p + [0.015, 0.01, 0.0]  # the pinch, below
        out.append(_Frame(i, j))
    return out


def test_the_gripper_comes_down_as_the_palm_does(tmp_path):
    """The gripper follows the palm's tilt, not the fingers' direction: a hand
    taking a thing from above keeps its fingers level, and the gripper came
    in level beside the object instead of down onto it."""
    from test_embodiment import SO101_URDF

    from dfpipeline.embodiment.robot import SO101_PROFILE, load

    arm = load(SO101_URDF, SO101_PROFILE, "SO-101")
    path = retarget_to_urdf(arm, {"": _palm_down_hand(range(90))}, TABLE, {"": None}, tmp_path, "SO-101")
    action = [r for r in json.loads(path.read_text())["trajectory"] if r["phase"] == "action"]
    down = [np.degrees(np.arccos(arm.approach(arm.fk(np.array(r["arm"]["joints_rad"][:5]))) @ [0, 0, -1]))
            for r in action]
    assert np.median(down) < 20, f"the gripper points {np.median(down):.0f} deg from straight down"
    assert np.median([r["position_error_m"] for r in action]) < 0.01


def test_the_gripper_turns_as_the_hand_turns(tmp_path):
    """An operator turns their wrist to line up with a handle: the gripper's
    roll turns with it. It read the thumb-to-index line, which hides behind
    the object and wanders, and barely turned (or spun to and fro)."""
    from test_embodiment import SO101_URDF
    from test_two_hands import _Frame

    from dfpipeline.embodiment.robot import SO101_PROFILE, load

    turn = np.radians(60)
    rng = np.random.default_rng(5)

    def turned(frames):
        out = []
        for f in frames:
            a = turn * f.frame_idx / 89                      # 0 -> 60 deg about the vertical
            c, s = np.cos(a), np.sin(a)
            centre = f.joints_world[8].copy()
            rel = f.joints_world - centre
            j = centre + np.column_stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1], rel[:, 2]])
            # The tips are behind the object: their line wanders, the palm's does not.
            r = rng.uniform(0, np.pi)
            pinch = 0.5 * (j[4] + j[8])
            j[4], j[8] = pinch - 0.015 * np.array([np.cos(r), np.sin(r), 0]), pinch + 0.015 * np.array([np.cos(r), np.sin(r), 0])
            out.append(_Frame(f.frame_idx, j))
        return out

    arm = load(SO101_URDF, SO101_PROFILE, "SO-101")
    path = retarget_to_urdf(arm, {"": turned(_palm_down_hand(range(90)))}, TABLE, {"": None}, tmp_path, "SO-101")
    action = [r for r in json.loads(path.read_text())["trajectory"] if r["phase"] == "action"]
    # Where the jaws point across the table, as the robot's own joints put them.
    jaw = [arm.jaw(arm.fk(np.array(r["arm"]["joints_rad"][:5]))) for r in action]
    yaw = np.degrees(np.unwrap(2 * np.arctan2([j[1] for j in jaw], [j[0] for j in jaw])) / 2)
    # The camera looks down: its turn one way is the robot's the other (up is -z).
    hand = -np.linspace(0, 60, len(yaw))
    follows = np.corrcoef(yaw, hand)[0, 1]
    assert follows > 0.9, f"the jaws follow the hand's turn with r = {follows:.2f}"
    assert np.ptp(yaw) > 40, f"the jaws turned {np.ptp(yaw):.0f} deg for a 60 deg turn of the hand"
    assert np.median([r["position_error_m"] for r in action]) < 0.01


def _handover(frames: int = 120) -> dict[str, list]:
    """Two hands 30 cm apart that come together over one point, and hold it:
    the left on the person's left (camera -x here; the robot's +y)."""
    from test_two_hands import _Frame

    rng = np.random.default_rng(3)
    shape = rng.normal(0, 0.02, (21, 3))
    shape[0] = [0, 0.08, 0]
    out = {"left": [], "right": []}
    for i in range(frames):
        t = min(1.0, i / 90)
        for side, x0 in (("left", -0.15), ("right", 0.15)):
            p = np.array([x0 * (1 - t), 0.0, 0.5])
            j = shape + p
            j[4], j[8] = p + [-0.02, -0.02, 0], p + [0.02, -0.02, 0]
            out[side].append(_Frame(i, j))
    return out


@pytest.mark.parametrize("which", ["pair", "two_arm"])
def test_a_handover_brings_the_two_grippers_together(tmp_path, which):
    """Each hand was centred on its own arm: when the hands met over a cup,
    each gripper stayed in front of its own arm, the pair's width apart. Both
    hands now share one frame, so where they meet, the grippers meet."""
    from test_embodiment import SO101_URDF

    from dfpipeline.embodiment.robot import SO101_PROFILE, load

    robot = load(SO101_URDF, SO101_PROFILE, "SO-101") if which == "pair" else Robot(BI_ARM6, BI_ARM6_PROFILE, "BiArm6")
    path = retarget_to_urdf(robot, _handover(), TABLE, {"left": None, "right": None}, tmp_path, "pair")
    rows = [r for r in json.loads(path.read_text())["trajectory"] if r["phase"] == "action"]
    # A pair stands side by side 40 cm apart (the default); a two-arm URDF's
    # arms share its base frame.
    half = 0.2 if which == "pair" else 0.0
    bases = {"left": np.array([0.0, half]), "right": np.array([0.0, -half])}

    def at(r, side):  # the gripper, in the shared frame
        return np.array(r[side]["ee_pos"]) + np.r_[bases[side], 0.0]

    start = np.linalg.norm(at(rows[0], "left")[:2] - at(rows[0], "right")[:2])
    met = np.median([np.linalg.norm(at(r, "left") - at(r, "right")) for r in rows[-20:]])
    assert met < 0.03, f"at the handover the grippers are {100 * met:.1f} cm apart"
    assert start > 0.1, "and they started apart, as the hands did"
    assert at(rows[0], "left")[1] > at(rows[0], "right")[1], "the left hand on the left"
