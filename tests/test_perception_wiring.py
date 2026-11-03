"""The hand-tracking models wired in: which detector, one clip at a time on the
GPU, the depth model's scale reaching the hand, and the bundle saying how its
distances were made (docs/HAND2ROBOT_2.md). No model is loaded."""

import numpy as np
import pytest

from dfpipeline import scheduler
from dfpipeline.config import settings
from dfpipeline.retarget import hand_wilor, lens, pipeline, scene


def _wilor_hand(is_right=1.0):
    kp3 = np.random.default_rng(0).uniform(-0.05, 0.05, (21, 3)) + [0, 0, 0.4]
    return {
        "hand_bbox": [100.0, 120.0, 300.0, 330.0],
        "is_right": is_right,
        "wilor_preds": {
            "pred_keypoints_3d": kp3[None],
            "pred_keypoints_2d": (kp3[:, :2] * 600 + 300)[None],
        },
    }


def test_a_wilor_hand_is_a_hand_detection_like_mediapipes():
    d = hand_wilor.to_detection(_wilor_hand(), 1280, 720)
    assert d.hand_side == "right"
    assert d.keypoints_2d.shape == (21, 2) and d.landmarks_world.shape == (21, 3)
    np.testing.assert_allclose(d.landmarks_world[0], 0.0)   # root-relative: PnP places it
    assert hand_wilor.to_detection(_wilor_hand(0.0), 1280, 720).hand_side == "left"


def test_an_incomplete_wilor_hand_is_dropped():
    broken = _wilor_hand()
    broken["wilor_preds"]["pred_keypoints_3d"] = np.zeros((1, 10, 3))
    assert hand_wilor.to_detection(broken, 1280, 720) is None
    assert hand_wilor.to_detection({}, 1280, 720) is None


def test_mediapipe_stays_the_default_detector(monkeypatch):
    monkeypatch.setattr(settings, "retarget_hand_model", "mediapipe")
    from dfpipeline.retarget.hand_detection import HandDetector

    monkeypatch.setattr(HandDetector, "__init__", lambda self: None)
    assert isinstance(hand_wilor.make_detector(), HandDetector)


@pytest.mark.parametrize("hand,depth,track", [
    ("mediapipe", "off", 2),
    ("wilor", "off", 1),
    ("mediapipe", "moge-2-vitb", 1),
])
def test_with_models_hand_tracking_takes_one_clip_at_a_time(monkeypatch, hand, depth, track):
    """Each worker process loads its own copy of every model: two at once is
    twice the GPU memory, and the GPU is the AI model's first."""
    monkeypatch.setattr(settings, "retarget_hand_model", hand)
    monkeypatch.setattr(settings, "retarget_depth_model", depth)
    monkeypatch.setattr(settings, "track_concurrency", 2)
    assert dict(scheduler.groups())["track"] == track
    cmd = scheduler.command("track", track)
    assert any(c.startswith("--max-memory-per-child") for c in cmd) == (track == 1)


class _Recon:
    def __init__(self, z):
        self.joints_camera = np.array([[0.01 * i, 0.0, z] for i in range(21)], dtype=np.float32)
        self.wrist_camera = self.joints_camera[0].copy()


def test_the_measured_scale_moves_every_reconstruction(monkeypatch):
    recons = [_Recon(0.4), _Recon(0.42)]
    seen = {0: [(np.zeros((21, 2)), recons[0])], 5: [(np.zeros((21, 2)), recons[1])]}
    found = scene.Scene(model="fake", hand_scale=1.2, table=scene.Plane(np.array([0, 0.5, 0.86]), 0.7, 0.9))
    monkeypatch.setattr(scene, "measure", lambda *a, **k: found)
    got = pipeline._measure_scene(None, lens.guess(640, 480), None, 10, seen, {})
    assert got is found
    assert recons[0].joints_camera[0, 2] == pytest.approx(0.48)
    assert recons[1].wrist_camera[2] == pytest.approx(0.504)
    assert pipeline._plane_distance(found, recons) == pytest.approx(0.7)


def test_a_failing_depth_model_leaves_the_hand_as_it_was(monkeypatch):
    recon = _Recon(0.4)

    def broken(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(scene, "measure", broken)
    assert pipeline._measure_scene(None, lens.guess(640, 480), None, 10, {0: [(None, recon)]}, {}) is None
    assert recon.joints_camera[0, 2] == pytest.approx(0.4)
    # And the table is estimated from the hand, as before.
    assert pipeline._plane_distance(None, [recon]) == pytest.approx(0.4)


def test_the_bundle_says_how_its_distances_were_made():
    without = pipeline._perception(lens.guess(1280, 720), None, None)
    assert without["lens_source"] == "guess" and without["depth_model"] == "off"
    found = scene.Scene(model="moge-2-vitb", frames_used=12, hand_scale=1.1)
    phone = lens.from_hfov(70.0, 1280, 720, "phone")
    got = pipeline._perception(phone, np.array([0.0, -1.0, 0.0]), found)
    assert got["lens_source"] == "phone" and got["up_known"] is True
    assert got["depth_model"] == "moge-2-vitb" and got["hand_scale"] == 1.1
