"""The hand-tracking models wired in: WiLoR the only detector, one clip at a
time on the GPU, the depth model's scale reaching the hand (and its failure
failing the clip), and the bundle saying how its distances were made
(docs/HAND2ROBOT_2.md). No model is loaded."""

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


def test_wilor_is_the_only_detector(monkeypatch):
    monkeypatch.setattr(hand_wilor.WilorDetector, "__init__", lambda self: None)
    assert isinstance(hand_wilor.make_detector(), hand_wilor.WilorDetector)


def test_hand_tracking_takes_one_clip_at_a_time(monkeypatch):
    """Each worker process loads its own copy of every model: two at once is
    twice the GPU memory, and the GPU is the AI model's first."""
    monkeypatch.setattr(settings, "track_concurrency", 2)
    assert dict(scheduler.groups())["track"] == 1
    assert any(c.startswith("--max-memory-per-child") for c in scheduler.command("track", 1))


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


def test_a_failing_depth_model_fails_the_clip(monkeypatch):
    """No measuring without it: the guessed distances it replaced were wrong."""
    def broken(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(scene, "measure", broken)
    with pytest.raises(RuntimeError):
        pipeline._measure_scene(None, lens.guess(640, 480), None, 10, {0: [(None, _Recon(0.4))]}, {})


def test_without_a_table_the_hand_gives_its_distance():
    recon = _Recon(0.4)
    assert pipeline._plane_distance(scene.Scene(model="moge-2-vitb"), [recon]) == pytest.approx(0.4)


def test_the_bundle_says_how_its_distances_were_made():
    found = scene.Scene(model="moge-2-vitb", frames_used=12, hand_scale=1.1)
    phone = lens.from_hfov(70.0, 1280, 720, "phone")
    got = pipeline._perception(phone, np.array([0.0, -1.0, 0.0]), found)
    assert got["lens_source"] == "phone" and got["up_known"] is True
    assert got["depth_model"] == "moge-2-vitb" and got["hand_scale"] == 1.1
    assert got["hand_model"] == "wilor"


def _sphere(n=12):
    """A small closed mesh: vertices (V, 3) and triangles (F, 3)."""
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n, endpoint=False), indexing="ij")
    v = 0.05 * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
    idx = np.arange(n * 2 * n).reshape(n, 2 * n)
    f = [tri for i in range(n - 1) for j in range(2 * n) for tri in (
        [idx[i, j], idx[i + 1, j], idx[i + 1, (j + 1) % (2 * n)]],
        [idx[i, j], idx[i + 1, (j + 1) % (2 * n)], idx[i, (j + 1) % (2 * n)]])]
    return v, np.asarray(f)


def test_wilors_mesh_lands_where_its_own_joints_do():
    """Projected the way WiLoR projects its 2D joints: its focal length, the
    image centre, its translation."""
    verts, _ = _sphere()
    t, f = np.array([0.02, -0.01, 30.0]), 25000.0
    preds = {"pred_vertices": verts[None], "pred_cam_t_full": t[None], "scaled_focal_length": f}
    wrist = verts[0]
    uv, xyz = hand_wilor.mesh_of(preds, wrist, 1280, 720)
    expected = f * (verts[:, :2] + t[:2]) / (verts[:, 2:] + t[2]) + [640, 360]
    np.testing.assert_allclose(uv, expected, rtol=1e-5)
    np.testing.assert_allclose(xyz.astype(float), verts - wrist, atol=1e-3)
    assert hand_wilor.mesh_of({}, wrist, 1280, 720) is None   # older output: no mesh, still a hand


def test_a_hand_keeps_its_mesh_and_saves_it_beside_its_trajectory(tmp_path, monkeypatch):
    verts, faces = _sphere()
    found = _wilor_hand()
    found["wilor_preds"].update(pred_vertices=verts[None], pred_cam_t_full=np.array([[0, 0, 30.0]]),
                                scaled_focal_length=25000.0)
    d = hand_wilor.to_detection(found, 1280, 720)
    assert d.mesh_uv.shape == (len(verts), 2) and d.mesh_xyz.shape == (len(verts), 3)
    monkeypatch.setattr(hand_wilor, "faces", lambda: faces)
    hand_wilor.save_meshes(tmp_path / "left", {3: d, 7: d})
    saved = np.load(tmp_path / "left" / "mesh.npz")
    assert list(saved["frames"]) == [3, 7] and saved["uv"].shape == (2, len(verts), 2)
    assert saved["faces"].shape == faces.shape
