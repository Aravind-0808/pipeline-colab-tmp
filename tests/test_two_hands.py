"""Tasks done with both hands: each hand tracked on its own, each onto its own SO-101."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from dfpipeline.lerobot_builder import TWO_ARM_FEATURE_NAMES, _robot_type, feature_names
from dfpipeline.retarget import service as rs
from dfpipeline.retarget.hand_detection import HandDetection, TwoHandTracker, merge_passes
from dfpipeline.retarget.object_tracking import ObjectTracker
from dfpipeline.validation.vlm import BOTH_HANDS, _task_details

DIAG = 1000.0


def det(x: float, y: float, side: str = "left", conf: float = 0.9) -> HandDetection:
    kp = np.tile(np.array([x, y], np.float32), (21, 1))
    return HandDetection(hand_side=side, bbox=[x - 40, y - 40, x + 40, y + 40], confidence=conf, keypoints_2d=kp)


# ---- tracking --------------------------------------------------------------------------


def test_each_hand_keeps_its_own_track_through_a_handover():
    tracker = TwoHandTracker(frame_diag_px=DIAG)
    seen = []
    # The hands come together from both sides, pass the cup, and part again.
    # MediaPipe names both "Left" all along — the label is no help here.
    for t in range(40):
        gap = abs(20 - t) * 10 + 90  # closest at t=20, never on top of each other
        left, right = det(400 - gap / 2, 300), det(400 + gap / 2, 300)
        found = [right, left] if t % 2 else [left, right]  # in either order
        seen.append(tracker.select(found))
    assert tracker.sides() in (["left", "right"], ["right", "left"])
    left_slot = tracker.sides().index("left")
    xs = [frame[left_slot].keypoints_2d[0][0] for frame in seen if frame[left_slot] is not None]
    assert len(xs) == 40
    assert all(x < 400 for x in xs), "the left track never jumped onto the right hand"


def test_the_left_arm_is_the_hand_on_the_left_of_the_picture():
    tracker = TwoHandTracker(frame_diag_px=DIAG)
    for _ in range(5):
        # MediaPipe calls the hand on the right "Left": it assumes a mirrored selfie image.
        tracker.select([det(700, 300, "left"), det(200, 300, "right")])
    sides = tracker.sides()
    left_slot = sides.index("left")
    assert tracker.slots[left_slot].wrist_x[0] == 200


def test_one_hand_seen_twice_is_not_two_hands():
    tracker = TwoHandTracker(frame_diag_px=DIAG)
    chosen = tracker.select([det(300, 300), det(310, 305, "right", 0.5)])
    assert sum(d is not None for d in chosen) == 1


def test_a_hand_that_appears_later_gets_the_free_track():
    tracker = TwoHandTracker(frame_diag_px=DIAG)
    for _ in range(5):
        tracker.select([det(200, 300)])
    chosen = tracker.select([det(205, 300), det(800, 300)])
    assert all(d is not None for d in chosen)


def test_the_reverse_pass_adds_the_hand_the_forward_pass_missed():
    forward = [[det(200, 300)], []]
    backward = [[det(202, 301), det(800, 300)], [det(500, 300)]]
    merged = merge_passes(forward, backward, DIAG)
    assert [len(f) for f in merged] == [2, 1]  # the same hand is not added twice


def test_either_hand_on_the_object_is_a_pinch_on_it():
    box = [100, 100, 200, 200]
    assert ObjectTracker._pinch_on(box, np.array([[500, 500], [150, 150]]))
    assert not ObjectTracker._pinch_on(box, np.array([[500, 500], [600, 600]]))
    assert ObjectTracker._pinch_on(box, np.array([150, 150]))  # one hand, as before


# ---- the gate --------------------------------------------------------------------------


def _hand_dir(tmp_path, spans: dict[str, tuple[int, int, int] | None]):
    """A two-hand bundle: per side (first, last, reconstructed frames), None = never seen."""
    (tmp_path / "hand_stage.json").write_text(json.dumps(
        {"fps": 30.0, "total_frames": 300, "hands": ["left", "right"]}))
    for side, span in spans.items():
        if span is None:
            continue
        first, last, done = span
        (tmp_path / side).mkdir()
        (tmp_path / side / "trajectory.json").write_text(json.dumps({"metadata": {
            "total_frames": 300, "fps": 30.0, "reconstructed_frames": done,
            "hand_first_frame": first, "hand_last_frame": last}}))
    return tmp_path


def test_both_hands_well_tracked_pass(tmp_path):
    metrics = rs.read_hand_metrics(_hand_dir(tmp_path, {"left": (0, 299, 290), "right": (10, 299, 280)}))
    assert set(metrics["hands"]) == {"left", "right"}
    assert metrics["hand_coverage"] == min(m["hand_coverage"] for m in metrics["hands"].values())
    assert rs.hand_gate(metrics) is None


def test_a_badly_tracked_hand_is_named(tmp_path):
    metrics = rs.read_hand_metrics(_hand_dir(tmp_path, {"left": (0, 299, 290), "right": (0, 299, 100)}))
    reason = rs.hand_gate(metrics)
    assert reason and reason.startswith("The right hand was reconstructed in only")


def test_a_hand_never_seen_fails_the_clip(tmp_path):
    metrics = rs.read_hand_metrics(_hand_dir(tmp_path, {"left": (0, 299, 290), "right": None}))
    reason = rs.hand_gate(metrics)
    assert reason and reason.startswith("The right hand was followed for only 0.0s")


def test_one_hand_reads_as_it_always_has(tmp_path):
    (tmp_path / "hand_stage.json").write_text(json.dumps({"fps": 30.0, "total_frames": 300}))
    (tmp_path / "trajectory.json").write_text(json.dumps({"metadata": {
        "total_frames": 300, "reconstructed_frames": 100, "hand_first_frame": 0, "hand_last_frame": 299}}))
    metrics = rs.read_hand_metrics(tmp_path)
    assert "hands" not in metrics
    assert rs.hand_gate(metrics).startswith("The hand was reconstructed in only")


# ---- the arms --------------------------------------------------------------------------


class _Frame:
    def __init__(self, i, j):
        self.frame_idx, self.timestamp, self.joints_world = i, i / 30.0, j


def _hand(x0: float, frames: range) -> list:
    rng = np.random.default_rng(3)
    shape = rng.normal(0, 0.02, (21, 3))
    shape[0] = [0, 0.08, 0]
    out = []
    for i in frames:
        t = i / 120
        p = np.array([x0 + 0.1 * np.sin(np.pi * t), 0.04 * t, 0.5 - 0.06 * np.sin(np.pi * t)])
        j = shape + p
        j[4], j[8] = p + [-0.02, -0.02, 0], p + [0.02, -0.02, 0]
        out.append(_Frame(i, j))
    return out


@pytest.fixture(scope="module")
def pair():
    from dfpipeline.retarget.retarget_so101 import retarget_pair

    # The right hand comes in a second later and leaves a second later.
    return retarget_pair({"left": _hand(-0.1, range(90)), "right": _hand(0.1, range(30, 120))},
                         np.array([0, 0, 1.0]), 0.5)


def test_both_arms_are_on_one_clock(pair):
    _, frames = pair
    left, right = frames["left"], frames["right"]
    assert len(left) == len(right)
    assert [f.timestamp for f in left] == [f.timestamp for f in right]
    assert [f.frame_idx for f in left] == [f.frame_idx for f in right]


def test_an_arm_waits_while_its_hand_is_not_there_yet(pair):
    _, frames = pair
    action = [i for i, f in enumerate(frames["right"]) if f.phase == "action"]
    early = [frames["right"][i] for i in action if frames["right"][i].frame_idx < 30]
    assert early, "the right arm's samples before its hand came in"
    assert all(not f.tracked for f in early)
    assert np.allclose([f.q_arm for f in early], early[0].q_arm)


def test_the_trajectory_names_both_arms_as_lerobot_does(pair, tmp_path):
    from dfpipeline.retarget.retarget_so101 import export_pair

    arms, frames = pair
    out = json.loads(export_pair(arms, frames, tmp_path).read_text())
    assert out["arms"] == ["left", "right"]
    sample = next(r for r in out["trajectory"] if r["phase"] == "action")
    assert sorted(sample["action"]) == sorted(TWO_ARM_FEATURE_NAMES)
    assert sample["position_error_m"] == max(sample["left"]["position_error_m"], sample["right"]["position_error_m"])
    # Joints in -100 .. 100 of their range, the gripper 0 .. 100 — not degrees.
    joints = [v for k, v in sample["action"].items() if not k.endswith("gripper.pos")]
    assert all(-100 <= v <= 100 for v in joints)
    assert all(0 <= sample["action"][f"{s}_arm_gripper.pos"] <= 100 for s in ("left", "right"))


def test_a_two_arm_dataset_has_twelve_columns_and_the_bimanual_robot_type():
    # As a two-arm SO-101 dataset names them (e.g. Qiu-Xinchuan/so101_towel).
    assert len(feature_names(2)) == 12 and feature_names(2)[0] == "left_arm_shoulder_pan.pos"
    assert feature_names(2)[-1] == "right_arm_gripper.pos"
    assert len(feature_names(1)) == 6
    assert _robot_type("so101", 2) == "bi_so101_follower"
    assert _robot_type("so101") == "so101_follower"


# ---- the AI check ----------------------------------------------------------------------


def test_the_ai_check_is_told_both_hands_are_used():
    two = SimpleNamespace(title="Hand the cup over", description="Pass it", task_flow=None, hands=2)
    one = SimpleNamespace(title="Pick the cup", description="Pick it", task_flow=None)
    assert BOTH_HANDS in _task_details(two)
    assert BOTH_HANDS not in _task_details(one)


def test_a_two_hand_clip_is_filmed_from_a_stand(tmp_path, monkeypatch):
    """Both hands do the task, so neither holds the phone: the camera is taken
    as still, whatever its motion estimate would have said."""
    import cv2

    from dfpipeline.retarget import pipeline

    clip = tmp_path / "c.mp4"
    out = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(10):
        out.write(np.zeros((48, 64, 3), np.uint8))
    out.release()
    seen = {}

    def two_hands(*args):
        seen["static_camera"] = args[7]
        return 0

    from dfpipeline.retarget import scene

    monkeypatch.setattr(pipeline, "_run_two_hands", two_hands)
    monkeypatch.setattr(pipeline, "probe_stream", lambda path: (10, 1 / 3))
    monkeypatch.setattr(scene, "estimate_hfov", lambda *a, **k: 70.0)   # the depth model's lens
    assert pipeline.run_hand_stage(clip, tmp_path / "out", hands=2, static_camera=False) == 0
    assert seen["static_camera"] is True
