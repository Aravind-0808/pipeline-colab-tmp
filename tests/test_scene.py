"""Real distances from a depth model: the table and the hand's size
(dfpipeline/retarget/scene.py). The model is stood in for: these test the
geometry, and cost no GPU and no download."""

import numpy as np
import pytest

from dfpipeline.config import settings
from dfpipeline.retarget import scene

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])


def _table_depth(distance=0.6, tilt_deg=30.0, h=480, w=640):
    """The depth of a table plane seen by a camera tilted down by tilt_deg."""
    t = np.radians(tilt_deg)
    normal = np.array([0.0, np.cos(t), np.sin(t)])          # camera -> table
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], axis=-1)
    depth = distance / (rays @ normal)
    depth[(rays @ normal) <= 0.05] = np.nan                  # rays above the horizon
    return depth, normal


def test_the_table_plane_is_found_with_its_real_distance():
    depth, normal = _table_depth()
    plane = scene.fit_plane(scene.table_points(depth, K, []))
    assert plane is not None
    np.testing.assert_allclose(plane.normal, normal, atol=1e-3)
    assert plane.distance == pytest.approx(0.6, abs=1e-3)
    assert plane.inliers > 0.95


def test_up_keeps_a_wall_from_being_taken_for_the_table():
    depth, _ = _table_depth()
    up = np.array([0.0, 1.0, 0.0])          # this phone says up is down the picture: no table faces it
    assert scene.fit_plane(scene.table_points(depth, K, []), up=np.array([1.0, 0.0, 0.0])) is None
    assert scene.fit_plane(scene.table_points(depth, K, []), up=up) is not None


def test_the_hand_is_kept_out_of_the_table():
    depth, _ = _table_depth()
    points_all = scene.table_points(depth, K, [])
    points_masked = scene.table_points(depth, K, [[200, 200, 440, 480]])
    assert len(points_masked) < len(points_all)


def test_noise_on_the_table_does_not_move_it():
    depth, normal = _table_depth()
    rng = np.random.default_rng(1)
    noisy = depth + rng.normal(0, 0.003, depth.shape)
    noisy[300:340, 100:160] -= 0.2                            # a cup on the table
    plane = scene.fit_plane(scene.table_points(noisy, K, []))
    np.testing.assert_allclose(plane.normal, normal, atol=0.02)
    assert plane.distance == pytest.approx(0.6, abs=0.01)


def _hand(z):
    """A hand's 21 joints at depth z, with their pixels."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.05, 0.05, (21, 2))
    joints = np.column_stack([xy, np.full(21, z) + rng.uniform(-0.02, 0.02, 21)])
    pixels = joints[:, :2] / joints[:, 2:] * 600 + [320, 240]
    return pixels, joints


def test_a_hand_bigger_than_average_is_moved_back_to_where_it_is():
    """The reconstruction put the hand at 0.40 m; the depth there is 0.50 m:
    the hand is 1.25x the average, and so is everything about where it is."""
    pairs = []
    for _ in range(5):
        _, joints = _hand(0.40)
        measured = joints[:, 2] * 1.25 - scene.SKIN_TO_JOINT_M   # the skin, in front of the joint
        pairs.append((measured, joints[:, 2]))
    found = scene.hand_scale(pairs)
    assert found is not None
    assert found[0] == pytest.approx(1.25, abs=1e-6)
    assert found[1] == pytest.approx(0.0, abs=1e-6)


def test_one_frame_or_a_hidden_hand_is_not_a_measurement():
    _, joints = _hand(0.4)
    assert scene.hand_scale([(joints[:, 2], joints[:, 2])]) is None
    hidden = np.full(21, np.nan)
    assert scene.hand_scale([(hidden, joints[:, 2])] * 5) is None


def test_scaling_about_the_camera_keeps_the_hand_on_its_pixels():
    pixels, joints = _hand(0.4)
    scaled = joints * 1.3
    np.testing.assert_allclose(scaled[:, :2] / scaled[:, 2:] * 600 + [320, 240], pixels, atol=1e-9)


def test_depth_at_reads_a_small_window_and_nothing_off_the_frame():
    depth = np.full((10, 10), 0.5)
    depth[5, 5] = 0.1                                 # one stray pixel
    got = scene.depth_at(depth, np.array([[5.0, 5.0], [20.0, 3.0]]))
    assert got[0] == pytest.approx(0.5)
    assert np.isnan(got[1])


def test_frame_zero_is_always_sampled():
    assert scene.sample_frames(100, 4)[0] == 0
    assert scene.sample_frames(0, 4) == []
    assert len(scene.sample_frames(3, 12)) == 3


class FakeDepth:
    """A depth model that sees a table at 0.6 m and a hand 1.25x farther
    than its reconstruction put it."""

    name = "fake"

    def __init__(self, hand_pixels, hand_depth):
        self.hand_pixels, self.hand_depth = hand_pixels, hand_depth
        self.calls = 0

    def infer(self, rgb, hfov):
        self.calls += 1
        depth, _ = _table_depth(h=rgb.shape[0], w=rgb.shape[1])
        for (u, v), z in zip(self.hand_pixels.round().astype(int), self.hand_depth):
            depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3] = z
        return depth, hfov or 62.0


def test_a_clip_is_measured_on_a_few_frames(tmp_path, monkeypatch):
    import cv2

    video = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for _ in range(60):
        out.write(np.zeros((480, 640, 3), np.uint8))
    out.release()

    pixels, joints = _hand(0.40)
    model = FakeDepth(pixels, joints[:, 2] * 1.25 - scene.SKIN_TO_JOINT_M)
    monkeypatch.setattr(settings, "retarget_depth_frames", 6)
    hands = {i: [(pixels, joints)] for i in range(60)}
    found = scene.measure(video, K, 56.0, 60, hands, {}, None, model=model)
    assert model.calls == 6, "a dozen frames at most, never every frame"
    assert found.table is not None and found.table.distance == pytest.approx(0.6, abs=0.01)
    assert found.hand_scale == pytest.approx(1.25, abs=0.01)
    assert found.summary()["table_source"] == "depth"


def test_without_a_depth_model_nothing_is_measured(monkeypatch):
    monkeypatch.setattr(settings, "retarget_depth_model", "off")
    scene.depth_model.cache_clear()
    try:
        assert scene.depth_model() is None
        assert scene.measure(None, K, None, 10, {}, {}, None) is None
    finally:
        scene.depth_model.cache_clear()


def test_a_scale_no_hand_has_is_not_applied(tmp_path):
    import cv2

    video = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for _ in range(30):
        out.write(np.zeros((480, 640, 3), np.uint8))
    out.release()
    pixels, joints = _hand(0.40)
    model = FakeDepth(pixels, joints[:, 2] * 3.0)          # three times the size: a fault
    found = scene.measure(video, K, 56.0, 30, {i: [(pixels, joints)] for i in range(30)}, {}, None, model=model)
    assert found.hand_scale is None and found.scale_measured > 2
    assert any("not applied" in n for n in found.notes)
