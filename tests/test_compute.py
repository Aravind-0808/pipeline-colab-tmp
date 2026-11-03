"""The three computes, as the app server calls them: by name, with a spec, by protocol.

The machine-learning calls — MediaPipe, the IK solve, the dataset build — are
stood in, so these run without torch or MediaPipe. What
they pin down is everything around those calls: what each result says, that
it survives JSON, that the gates are decided here, and that a message from
another protocol version is refused.
"""

import io
import json
from pathlib import Path

import pytest

from dfpipeline import PROTOCOL_VERSION
from dfpipeline import compute as computes
from dfpipeline.retarget import service as rs
from dfpipeline.storage import LocalStorage, get_storage, use_storage


def run(name, spec):
    """As the Celery task runs it — and back through JSON, as Redis carries it."""
    return json.loads(json.dumps(computes.run(name, json.loads(json.dumps(spec)), PROTOCOL_VERSION)))


class Numberish:
    """Stands in for a numpy scalar: has .item(), is not JSON by itself."""

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


@pytest.fixture
def staged(tmp_path):
    """A clip in the store, as the app describes it (its key in the bucket)."""
    store = LocalStorage(str(tmp_path / "store"))
    store.put("p/t/clip.webm", io.BytesIO(b"bytes"))
    working = get_storage()
    use_storage(store)
    yield {"video_id": "v1", "path": "p/t/clip.webm", "size_bytes": 5, "kind": "s3", "bucket": "dataforge"}
    use_storage(working)


# ---------------------------------------------------------------- protocol


def test_a_message_from_another_protocol_version_is_refused():
    out = computes.run("track.hand", {}, PROTOCOL_VERSION + 1)
    assert out["ok"] is False
    assert "protocol mismatch" in out["error"]
    assert computes.run("track.hand", {}, None)["ok"] is False


def test_an_unknown_compute_is_refused():
    assert computes.run("no.such", {}, PROTOCOL_VERSION) == {
        "ok": False, "error": "No compute function 'no.such'"
    }


def test_a_crash_crosses_the_queue_as_text(monkeypatch):
    monkeypatch.setitem(computes._COMPUTE, "boom", lambda spec: 1 / 0)
    assert computes.run("boom", {}, PROTOCOL_VERSION) == {
        "ok": False, "error": "ZeroDivisionError: division by zero"
    }


# ---------------------------------------------------------------- track.hand


@pytest.fixture
def tracked(monkeypatch):
    stored = []

    def track_hand(source, work, hands=1, camera=None):
        assert Path(source).is_file(), "the clip should have been read from disk"
        hand = work / "hand"
        hand.mkdir()
        return {"hand_coverage": Numberish(0.95), "object_frames_fraction": 0.9}, hand

    monkeypatch.setattr(rs, "track_hand", track_hand)
    monkeypatch.setattr(rs, "store_bundle", lambda hand_dir, path: stored.append(path))
    return stored


def test_a_two_hand_task_is_tracked_with_both_hands(staged, monkeypatch):
    asked = []

    def track_hand(source, work, hands=1, camera=None):
        asked.append(hands)
        hand = work / "hand"
        hand.mkdir()
        return {"hand_coverage": 0.95}, hand

    monkeypatch.setattr(rs, "track_hand", track_hand)
    monkeypatch.setattr(rs, "store_bundle", lambda hand_dir, path: None)
    run("track.hand", {"video_id": "v1", "clip": staged, "bundle_path": "b", "hands": 2})
    run("track.hand", {"video_id": "v2", "clip": staged, "bundle_path": "b"})  # an older app: one hand
    assert asked == [2, 1]


def test_the_phones_camera_data_reaches_the_hand_stage(staged, monkeypatch):
    """The lens and gravity the phone sent (lens.py); an older app sends none,
    and something that is not an object is not passed on."""
    got = []

    def track_hand(source, work, hands=1, camera=None):
        got.append(camera)
        hand = work / "hand"
        hand.mkdir()
        return {"hand_coverage": 0.95}, hand

    monkeypatch.setattr(rs, "track_hand", track_hand)
    monkeypatch.setattr(rs, "store_bundle", lambda hand_dir, path: None)
    camera = {"schema": 1, "logical": {"focal_lengths_mm": [4.2]}}
    run("track.hand", {"video_id": "v1", "clip": staged, "bundle_path": "b", "camera": camera})
    run("track.hand", {"video_id": "v2", "clip": staged, "bundle_path": "b"})
    run("track.hand", {"video_id": "v3", "clip": staged, "bundle_path": "b", "camera": "junk"})
    assert got == [camera, None, None]


def test_a_tracked_clip_comes_back_with_its_numbers_and_the_gates_verdict(staged, tracked):
    out = run("track.hand", {"video_id": "v1", "clip": staged, "bundle_path": "tracks/x/v1.zip"})
    assert out["ok"], out
    value = out["value"]
    assert value["metrics"]["hand_coverage"] == 0.95  # numpy-like became a float
    assert value["bundle_path"] == "tracks/x/v1.zip"
    assert "overlay_path" not in value  # nothing drawn, nothing stored but the bundle
    assert value["reason"] is None  # the gate let it through
    assert tracked == ["tracks/x/v1.zip"]


def test_the_hand_gate_is_decided_here(staged, monkeypatch, tracked):
    monkeypatch.setattr(
        rs, "track_hand",
        lambda source, work, hands=1, camera=None: ({"hand_coverage": 0.2, "object_frames_fraction": 0.9}, work),
    )
    out = run("track.hand", {"video_id": "v1", "clip": staged, "bundle_path": "b"})
    assert out["value"]["reason"]  # refused, with a reason the operator can read


def test_a_clip_the_tracker_cannot_read_is_an_answer_not_a_crash(staged, monkeypatch):
    def unreadable(source, work, hands=1, camera=None):
        raise rs.RetargetFailed("no hand found in any frame")

    monkeypatch.setattr(rs, "track_hand", unreadable)
    out = run("track.hand", {"video_id": "v1", "clip": staged, "bundle_path": "b"})
    assert out == {"ok": True, "value": {"failed": "no hand found in any frame"}}


# ------------------------------------------------------------- retarget.solve


def test_a_retarget_comes_back_with_its_trajectory_and_the_gates_verdict(monkeypatch):
    stored = []
    monkeypatch.setattr(rs, "fetch_bundle", lambda path, work: work)
    monkeypatch.setattr(rs, "retarget_bundle",
                        lambda h, c, w: ({"reachable_fraction": 0.5, "calibrated": True}, w / "t"))
    monkeypatch.setattr(rs, "store", lambda trajectory, path: stored.append(path))
    out = run("retarget.solve", {"bundle_path": "b", "calibration": {"k": 1}, "store_path": "retargets/x/r/v.json.gz"})
    value = out["value"]
    assert value["trajectory_path"] == "retargets/x/r/v.json.gz"
    assert "reach" in (value["reason"] or "").lower()  # 50% reachable: refused
    assert stored == ["retargets/x/r/v.json.gz"]


def test_a_bundle_not_yet_on_the_hub_is_reported_as_pending(monkeypatch):
    def not_there(path, work):
        raise FileNotFoundError(path)

    monkeypatch.setattr(rs, "fetch_bundle", not_there)
    out = run("retarget.solve", {"bundle_path": "b", "calibration": None, "store_path": "s"})
    assert out == {"ok": True, "value": {"bundle_pending": True}}


# --------------------------------------------------------------- export.build


def test_a_build_reports_progress_to_server_a_and_returns_the_dataset(monkeypatch, tmp_path):
    import dfpipeline.lerobot_builder as builder
    import dfpipeline.worker as worker

    progress = []
    monkeypatch.setattr(worker, "send_progress", lambda *a: progress.append(a))
    monkeypatch.setattr(computes.settings, "export_data_dir", str(tmp_path))

    def build(payload, root, report):
        report(0.5, "episode 1/2")
        report(0.5, "episode 1/2")  # the same step again, within 5s: not sent twice
        return {"total_frames": Numberish(10), "lerobot_version": "x", "hub_url": "u",
                "commit_sha": "c", "root": str(Path(root)), "repo_id": payload["repo_id"]}

    monkeypatch.setattr(builder, "build_to_store", build)
    out = run("export.build", {"payload": {"export_id": "e1", "repo_id": "someone/ds", "episodes": []}})
    assert out["value"]["total_frames"] == 10
    assert progress == [("e1", 0.5, "episode 1/2")]


def test_a_robot_build_waits_for_trajectories_not_yet_on_the_hub(staged, monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError

    def not_on_the_hub(path):
        # What the Hub says about a file another machine has not committed yet.
        raise EntryNotFoundError(f"404: {path}")

    monkeypatch.setattr(get_storage(), "fetch_local", not_on_the_hub)
    out = run("export.build", {"payload": {
        "export_id": "e2", "repo_id": "someone/ds", "robot_id": "r1",
        "episodes": [{"trajectory_path": "retargets/x/r1/never-made.json.gz"}],
    }})
    assert out["value"] == {"pending": 1}


def test_numbers_from_the_ml_stack_become_plain_json():
    value = computes.plain({"a": Numberish(1.5), "b": [Numberish(2), Path("/x/y")], "c": None})
    assert value == {"a": 1.5, "b": [2, "/x/y"], "c": None}

