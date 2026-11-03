"""The LeRobot builder's contract with the library.

Every export died on its first line: the builder imported
`lerobot.common.datasets`, which 0.6 renamed to `lerobot.datasets`. Nothing
caught it because nothing imported the builder — the failure only happened
inside a worker, on a path no test walked. Zero exports had ever succeeded.

These check the handful of library symbols the builder actually depends on, so
the next rename is a failing test rather than a dead queue job.
"""

import json
from pathlib import Path

import pytest

lerobot = pytest.importorskip("lerobot", reason="needs the 'ml' extra")


def test_the_dataset_class_can_be_found():
    from dfpipeline.lerobot_builder import _dataset_class

    assert _dataset_class().__name__ == "LeRobotDataset"


@pytest.mark.parametrize("method", ["create", "add_frame", "save_episode", "finalize", "push_to_hub"])
def test_the_builder_calls_methods_that_exist(method):
    from dfpipeline.lerobot_builder import _dataset_class

    assert hasattr(_dataset_class(), method), f"the builder calls {method}()"


def test_create_still_takes_the_arguments_the_builder_passes():
    """repo_id, fps and features positionally, then root and use_videos."""
    import inspect

    from dfpipeline.lerobot_builder import _dataset_class

    params = inspect.signature(_dataset_class().create).parameters
    for name in ("repo_id", "fps", "features", "root", "use_videos"):
        assert name in params, f"create() no longer accepts {name}"


def test_a_frame_carries_the_task():
    """The instruction a policy is conditioned on. LeRobot reads it off every
    frame and builds meta/tasks.parquet and the episode's tasks from it, so a
    frame without one produces a dataset nothing can train against."""
    import inspect

    from dfpipeline.lerobot_builder import _dataset_class

    assert "frame" in inspect.signature(_dataset_class().add_frame).parameters


def test_a_build_directory_is_cleared_once_it_is_on_the_hub(tmp_path):
    """The setting to keep it existed and nothing read it, so every export left
    its whole dataset behind — one 47-episode build was 284MB."""
    from dfpipeline.compute import discard_build as _discard_build

    root = tmp_path / "robocollect__set-v0.1.0"
    (root / "videos" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "chunk-000" / "file-000.mp4").write_bytes(b"x" * 1024)
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text("{}")

    assert _discard_build(root) is True
    assert not root.exists()


def test_clearing_a_build_that_is_not_there_is_not_an_error(tmp_path):
    from dfpipeline.compute import discard_build as _discard_build

    assert _discard_build(tmp_path / "never-built") is False


def test_a_locked_build_directory_does_not_fail_the_export(tmp_path, monkeypatch):
    """Cleanup is tidying, not the job. A dataset already safely on the Hub
    must not be reported as failed because a file was in use."""
    import shutil

    from dfpipeline.compute import discard_build as _discard_build

    root = tmp_path / "locked"
    root.mkdir()

    def refuse(*_a, **_k):
        raise OSError("file is in use by another process")

    monkeypatch.setattr(shutil, "rmtree", refuse)
    assert _discard_build(root) is False
    assert root.exists()



# --- building into the store ------------------------------------------------
#
# A build reads each clip from the store (sources.materialize), writes the
# dataset locally and copies it into the store at payload["store_prefix"].
# Nothing goes to Hugging Face during a build: that is `publish`, later, when
# an admin asks. The video work and the clip reads are stubbed; LeRobot itself
# is real.


def _clip(name: str) -> dict:
    return {"kind": "s3", "path": f"raw/p/t/{name}.mp4"}


def _world(monkeypatch, tmp_path, frames: int = 4, broken=("broken", "bad")):
    """Stub the clip reads and the video work, use a folder as the store, and
    make any call to the Hub a failure. Returns the builder module."""
    import numpy as np

    from dfpipeline import lerobot_builder as B
    from dfpipeline.storage import LocalStorage, use_storage

    store = LocalStorage(str(tmp_path / "store"))
    use_storage(store)
    monkeypatch.setattr(B.settings, "export_recover_pause_sec", 0)
    # The fake clips are a few frames long; the minimum has its own tests.
    monkeypatch.setattr(B.settings, "export_min_episode_frames", 1)

    def materialize(spec, work_dir=None):
        if any(b in spec["path"] for b in broken):
            raise RuntimeError("500 Server Error: clip is unavailable")
        path = Path(work_dir or tmp_path) / f"{abs(hash(spec['path'])) % 100_000}.mp4"
        path.write_bytes(b"not really a video")
        return path

    monkeypatch.setattr(B.sources, "materialize", materialize)
    monkeypatch.setattr(B.video, "probe_size", lambda p: (32, 32))
    monkeypatch.setattr(
        B.video, "iter_frames",
        lambda src, fps, size, work: (np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(frames)),
    )

    class NoHub:
        def __init__(self, *a, **k):
            pytest.fail("a build must not touch Hugging Face; publishing is separate")

    monkeypatch.setattr("huggingface_hub.HfApi", NoHub)
    return B


@pytest.fixture(autouse=True)
def _forget_the_store():
    yield
    from dfpipeline.storage import use_storage

    use_storage(None)


def _payload(name: str, episodes: list[dict], **extra) -> dict:
    return {"fps": 30, "image_key": "observation.images.top", "repo_id": f"local/{name}",
            "store_prefix": f"datasets/{name}/v1", "episodes": episodes, **extra}


def test_a_build_goes_into_the_store_and_never_to_the_hub(tmp_path, monkeypatch):
    B = _world(monkeypatch, tmp_path)

    result = B.build_to_store(
        _payload("stored", [{"task": "Pick up the cup", "clip": _clip("a")}]), tmp_path / "ds"
    )

    assert result["store_prefix"] == "datasets/stored/v1/"
    assert result["files"] > 0
    stored = tmp_path / "store" / "datasets" / "stored" / "v1"
    assert (stored / "meta" / "info.json").is_file(), "the finished dataset, in the store"


def test_a_clip_that_will_not_load_does_not_cost_the_whole_export(tmp_path, monkeypatch):
    """One 500 must not throw away the other forty-nine episodes. Rebuilding a
    large dataset because one clip failed is hours of work for one bad clip."""
    B = _world(monkeypatch, tmp_path)

    result = B.build_to_store(
        _payload("skip", [
            {"task": "Pick up the cup", "clip": _clip("good-1")},
            {"task": "Pick up the cup", "clip": _clip("broken")},
            {"task": "Pick up the cup", "clip": _clip("good-2")},
        ]),
        tmp_path / "ds",
    )

    assert result["files"] > 0, "the two good episodes should still be stored"
    assert result["total_frames"] == 8, "two episodes of four frames, not three"


def test_an_export_where_every_clip_fails_is_an_error(tmp_path, monkeypatch):
    """Skipping is for the odd bad clip. A dataset with nothing in it is not a
    success, and storing an empty one would be worse than failing."""
    B = _world(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="No episode could be built"):
        B.build_to_store(_payload("none", [{"task": "t", "clip": _clip("broken")}]), tmp_path / "ds")
    assert not (tmp_path / "store" / "datasets").exists(), "nothing stored"


def test_a_bad_clip_at_the_front_does_not_kill_the_export(tmp_path, monkeypatch):
    """The frame size comes from the first readable clip, not from episode
    zero. Taking it from episode zero regardless put that one read outside the
    loop's tolerance, so one bad clip at the front killed an export whose
    other forty-nine were fine."""
    B = _world(monkeypatch, tmp_path)

    result = B.build_to_store(
        _payload("front", [
            {"task": "t", "clip": _clip("broken")},
            {"task": "t", "clip": _clip("good-1")},
            {"task": "t", "clip": _clip("good-2")},
        ]),
        tmp_path / "ds-front",
    )

    assert result["total_frames"] == 8, "the two readable episodes, not one and not none"


def test_a_fresh_build_preserves_and_avoids_what_a_failed_one_left(tmp_path, monkeypatch):
    """A fresh export must not overwrite or delete an earlier build."""
    B = _world(monkeypatch, tmp_path, frames=2)

    root = tmp_path / "robocollect__set-v0.1.0"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text('{"total_frames": 1}')
    (root / "leftover.bin").write_bytes(b"from the run before")

    result = B.build_to_store(
        _payload("fresh", [{"task": "t", "clip": _clip("a")}], resume=False), root
    )

    assert result["total_frames"] == 2, "the new build, not the one left behind"
    assert (root / "leftover.bin").exists(), "a different worker may still own the old build"
    assert Path(result["root"]) != root


def test_resuming_keeps_what_the_failed_build_made(tmp_path, monkeypatch):
    """The whole point of resume is not paying to encode it all again: a
    finished build with the same inputs is stored as it is."""
    B = _world(monkeypatch, tmp_path)
    monkeypatch.setattr(
        B.sources, "materialize", lambda *a, **k: pytest.fail("resume must not read the clips again")
    )

    root = tmp_path / "robocollect__resume-v0.1.0"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text('{"total_frames": 4242}')
    payload = _payload("resume", [{"task": "t", "clip": _clip("a")}], resume=True)
    (root / ".export-complete.json").write_text(
        json.dumps({"fingerprint": B._build_fingerprint(payload)}), encoding="utf-8"
    )

    result = B.build_to_store(payload, root)

    assert result["total_frames"] == 4242, "the build that was already there"
    assert (tmp_path / "store" / "datasets" / "resume" / "v1" / "meta" / "info.json").is_file()


def test_a_locked_leftover_sends_the_build_next_door(tmp_path, monkeypatch):
    """On Windows a worker that built a dataset earlier still has its parquet
    memory-mapped, and an open file cannot be deleted. Refusing to export until
    someone restarts the worker is a worse answer than building beside it."""
    import shutil

    from dfpipeline.lerobot_builder import _clear_or_sidestep

    root = tmp_path / "robocollect__set-v0.1.0"
    root.mkdir()
    (root / "held-open.parquet").write_bytes(b"x")

    monkeypatch.setattr(
        shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(OSError("in use"))
    )

    chosen = _clear_or_sidestep(root)
    assert chosen != root, "a locked directory should not be built into"
    assert chosen.parent.name.startswith(root.name), "it should be recognisably the same export"
    assert root.exists(), "the locked one is left for whoever holds it"


def test_fresh_builds_have_separate_reserved_workspaces(tmp_path):
    from dfpipeline.lerobot_builder import _clear_or_sidestep

    root = tmp_path / "robocollect__set-v0.1.0"
    root.mkdir()
    (root / "old.bin").write_bytes(b"x")

    first = _clear_or_sidestep(root)
    second = _clear_or_sidestep(root)
    assert first != second
    assert first.parent.is_dir() and second.parent.is_dir()
    assert not first.exists() and not second.exists()
    assert (root / "old.bin").read_bytes() == b"x"



def test_an_export_stops_once_the_failures_are_clearly_not_the_clips(tmp_path, monkeypatch):
    """Skipping is for the odd bad clip. One run failed every episode the same
    way and worked through 52 of 349 before anyone noticed — hours of
    downloading and 117MB of half-built episodes to prove what the first five
    already showed."""
    B = _world(monkeypatch, tmp_path)
    monkeypatch.setattr(B.settings, "export_give_up_after", 5)
    attempts = {"n": 0}
    read = B.sources.materialize

    def counting(spec, work_dir=None):
        attempts["n"] += 1
        return read(spec, work_dir)

    def always_breaks(src, fps, size, work):
        raise RuntimeError("[WinError 3] The system cannot find the path specified")

    monkeypatch.setattr(B.sources, "materialize", counting)
    monkeypatch.setattr(B.video, "iter_frames", always_breaks)

    with pytest.raises(RuntimeError, match="failed in a row"):
        B.build_to_store(
            _payload("doomed", [{"task": "t", "clip": _clip(f"c{i}")} for i in range(349)]),
            tmp_path / "ds-doomed",
        )

    assert attempts["n"] <= 6, f"stopped after {attempts['n']} episodes, not 349"


def test_the_odd_bad_clip_still_does_not_stop_an_export(tmp_path, monkeypatch):
    """The give-up rule must not undo skipping: failures scattered among good
    episodes are what skipping is for."""
    B = _world(monkeypatch, tmp_path, frames=2)
    monkeypatch.setattr(B.settings, "export_give_up_after", 5)

    # A bad one every third episode — never five in a row.
    episodes = [
        {"task": "t", "clip": _clip(f"{'bad' if i % 3 == 2 else 'good'}-{i}")} for i in range(12)
    ]
    result = B.build_to_store(_payload("scattered", episodes), tmp_path / "ds-scattered")

    assert result["files"] > 0, "eight good episodes should still be stored"
    assert result["total_frames"] == 16


@pytest.mark.parametrize("complete,matching", [(False, True), (True, False), (True, True)])
def test_resume_finds_only_complete_matching_builds(tmp_path, complete, matching):
    from dfpipeline import lerobot_builder as B

    root = tmp_path / "owner__dataset"
    built = B._clear_or_sidestep(root)
    (built / "meta").mkdir(parents=True)
    (built / "meta" / "info.json").write_text('{"total_frames": 2}')
    if complete:
        (built / ".export-complete.json").write_text(
            json.dumps({"fingerprint": "matching" if matching else "other"})
        )
    assert B._find_completed_build(root, "matching") == (
        built if complete and matching else None
    )



def test_resume_fingerprint_ignores_expiring_signatures_but_checks_settings():
    """The clip's key in the store is what identifies it; a build with other
    settings is a different build."""
    from dfpipeline.lerobot_builder import _build_fingerprint

    payload = {"fps": 30, "image_key": "observation.images.top",
               "episodes": [{"task": "pick", "clip": {"kind": "s3", "path": "raw/p/t/a.mp4",
                                                      "url": "http://x/clip?token=old"}}]}
    original = _build_fingerprint(payload)
    payload["episodes"][0]["clip"]["url"] = "http://x/clip?token=new"
    assert _build_fingerprint(payload) == original
    payload["fps"] = 60
    assert _build_fingerprint(payload) != original


def _language(root: Path):
    import pandas as pd

    return pd.concat(pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet")))


def test_the_timeline_the_check_saw_is_written_onto_the_episodes(tmp_path, monkeypatch):
    """This was a second pass over the finished dataset asking the model a
    question every couple of seconds of video. The check already looks at a
    timestamped grid of every clip, so the timeline arrives with the verdict
    and the build only has to write it down."""
    import pandas as pd

    B = _world(monkeypatch, tmp_path, frames=6)  # 0.2 s at 30 fps

    steps = [
        {"name": "reach for the cup", "start": 0.0, "end": 0.1},
        {"name": "grasp the cup", "start": 0.1, "end": 0.2},
    ]
    result = B.build_to_store(
        _payload("annotated", [
            {"task": "Pick up the cup", "clip": _clip("a"), "subtasks": steps},
            {"task": "Pick up the cup", "clip": _clip("b"), "subtasks": steps},
        ]),
        tmp_path / "ds-annotated",
    )

    assert result["annotation"]["annotated"] is True
    assert result["annotation"]["episodes"] == 2
    # Written by us, from the check — not by a second pass over the video.
    assert result["annotation"]["source"] == "validation"
    built = Path(result["root"])

    # LeRobot 0.6's format, the one its annotation viewer reads: every frame
    # carries the episode's subtask rows, each stamped with its first frame.
    info = json.loads((built / "meta" / "info.json").read_text())
    assert info["features"]["language_persistent"]["dtype"] == "language"
    data = _language(built)
    rows = list(data["language_persistent"].iloc[0])
    assert [(r["style"], r["content"]) for r in rows] == [
        ("subtask", "reach for the cup"), ("subtask", "grasp the cup")]
    assert [round(r["timestamp"], 3) for r in rows] == [0.0, 0.1]
    assert all(len(v) == 2 for v in data["language_persistent"]), "on every frame"
    assert all(len(v) == 0 for v in data["language_events"])

    # The older per-episode lists, kept for the tools that read those.
    files = list((built / "meta" / "episodes").rglob("*.parquet"))
    frame = pd.read_parquet(files[0])
    for column in ("sparse_subtask_names", "subtask_names"):
        assert column in frame.columns, f"LeRobot's readers look for {column}"
    assert list(frame.at[0, "sparse_subtask_names"]) == ["reach for the cup", "grasp the cup"]
    assert list(frame.at[0, "sparse_subtask_start_frames"]) == [0, 3]
    assert list(frame.at[0, "sparse_subtask_end_frames"]) == [3, 6]
    assert list(frame.at[0, "subtask_names"]) == ["reach for the cup", "grasp the cup"]


def test_steps_are_placed_on_the_robots_clock_not_the_videos():
    """A robot episode runs slower than the human, so a video frame is repeated
    for several rows. A step that starts at second 1 of the clip starts on the
    first row taken from second 1 — not on row fps x 1."""
    from dfpipeline.lerobot_builder import _step_rows

    # Each clip second repeated twice: the robot took twice as long.
    clip_times = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5]
    steps = [{"name": "reach", "start": 0.0, "end": 1.0}, {"name": "grasp", "start": 1.0, "end": 2.0}]
    assert _step_rows(steps, clip_times) == [("reach", 0, 3), ("grasp", 4, 7)]


def test_a_step_no_row_falls_in_is_left_out():
    from dfpipeline.lerobot_builder import _step_rows

    clip_times = [0.0, 0.1, 0.2]
    steps = [{"name": "seen", "start": 0.0, "end": 0.3}, {"name": "after the end", "start": 5.0, "end": 6.0}]
    assert _step_rows(steps, clip_times) == [("seen", 0, 2)]


def test_a_dataset_without_timelines_still_builds(tmp_path, monkeypatch):
    """Clips checked before this existed have no steps on them, and a dataset
    without a timeline is still a dataset."""
    B = _world(monkeypatch, tmp_path, frames=6)

    result = B.build_to_store(
        _payload("plain", [{"task": "Pick up the cup", "clip": _clip("a")}]), tmp_path / "ds-plain"
    )
    assert result["annotation"]["annotated"] is False
    assert result["total_frames"] == 6, "the episode is still built and stored"


def test_a_skipped_clip_does_not_shift_the_timelines(tmp_path, monkeypatch):
    """Episode indexes are handed out as clips are accepted, so a clip that
    fails must not leave every timeline after it attached to the wrong
    episode — the worst kind of wrong, because it looks fine."""
    import pandas as pd

    B = _world(monkeypatch, tmp_path, frames=6)

    first = [{"name": "first clip step", "start": 0.0, "end": 1.0}]
    third = [{"name": "third clip step", "start": 0.0, "end": 1.0}]
    result = B.build_to_store(
        _payload("shifted", [
            {"task": "t", "clip": _clip("good-1"), "subtasks": first},
            {"task": "t", "clip": _clip("broken"), "subtasks": [
                {"name": "never built", "start": 0.0, "end": 1.0}]},
            {"task": "t", "clip": _clip("good-2"), "subtasks": third},
        ]),
        tmp_path / "ds-shifted",
    )

    files = list((Path(result["root"]) / "meta" / "episodes").rglob("*.parquet"))
    frame = pd.read_parquet(files[0])
    assert list(frame.at[0, "sparse_subtask_names"]) == ["first clip step"]
    # The second built episode is the third clip, and carries the third's steps.
    assert list(frame.at[1, "sparse_subtask_names"]) == ["third clip step"]


# --- adding clips: the next version is merged onto the previous one ------------


def test_adding_clips_merges_onto_the_previous_version(tmp_path, monkeypatch):
    """Once a dataset is built, its raw clips are gone. The next version is the
    new clips only, merged onto the previous version — not a rebuild."""
    import pandas as pd

    B = _world(monkeypatch, tmp_path, frames=6)
    steps = [{"name": "reach", "start": 0.0, "end": 0.1}]

    v1 = B.build_to_store(
        _payload("grow", [
            {"task": "t", "clip": _clip("a"), "video_id": "a", "subtasks": steps},
            {"task": "t", "clip": _clip("b"), "video_id": "b"},
        ], store_prefix="datasets/grow/v1"),
        tmp_path / "ds-v1",
    )
    assert v1["episodes"] == 2 and v1["video_ids"] == ["a", "b"]

    # The new clip may be a different size: it is made at the previous version's.
    B.video.probe_size = lambda p: (48, 64)
    v2 = B.build_to_store(
        _payload("grow", [{"task": "t", "clip": _clip("c"), "video_id": "c", "subtasks": steps}],
                 store_prefix="datasets/grow/v2", base={"store_prefix": "datasets/grow/v1/"}),
        tmp_path / "ds-v2",
    )

    assert v2["episodes"] == 3, "two from the previous version, one new"
    assert v2["video_ids"] == ["c"], "only the new clip is reported as built"
    assert v2["total_frames"] == 18
    stored = tmp_path / "store" / "datasets" / "grow" / "v2"
    info = json.loads((stored / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    names = pd.concat(pd.read_parquet(p) for p in (stored / "meta" / "episodes").rglob("*.parquet"))
    assert list(names.sort_values("episode_index")["sparse_subtask_names"].map(
        lambda v: list(v) if v is not None else None)) == [["reach"], None, ["reach"]]
    data = _language(stored).sort_values(["episode_index", "frame_index"])
    per_episode = data.groupby("episode_index")["language_persistent"].first()
    assert [[r["content"] for r in v] for v in per_episode] == [["reach"], [], ["reach"]]


def test_a_new_version_with_a_timeline_merges_onto_one_without(tmp_path, monkeypatch):
    """Versions built before the language columns existed have none. The next
    version carries them, and the two still merge."""
    B = _world(monkeypatch, tmp_path, frames=6)
    B.build_to_store(
        _payload("old", [{"task": "t", "clip": _clip("a"), "video_id": "a"}],
                 store_prefix="datasets/old/v1"),
        tmp_path / "ds-v1",
    )
    v2 = B.build_to_store(
        _payload("old", [{"task": "t", "clip": _clip("b"), "video_id": "b",
                          "subtasks": [{"name": "reach", "start": 0.0, "end": 0.2}]}],
                 store_prefix="datasets/old/v2", base={"store_prefix": "datasets/old/v1/"}),
        tmp_path / "ds-v2",
    )
    assert v2["episodes"] == 2
    stored = tmp_path / "store" / "datasets" / "old" / "v2"
    data = _language(stored)
    assert "language_persistent" in data.columns


def test_a_previous_version_that_cannot_be_read_fails_the_build(tmp_path, monkeypatch):
    B = _world(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="previous version"):
        B.build_to_store(
            _payload("lost", [{"task": "t", "clip": _clip("a")}], base={"store_prefix": "datasets/none/"}),
            tmp_path / "ds",
        )


def _hub(monkeypatch, tags=(), comes_out_public=False, stays_public=False):
    """A fake Hugging Face: records what was asked of it; `tags` already exist.
    `comes_out_public`: a repo asked for as private is created public anyway;
    `stays_public`: and cannot be made private."""
    calls = {}
    state = {"private": None}

    class Hub:
        def __init__(self, token):
            pass

        def repo_info(self, repo_id, repo_type):
            return type("Info", (), {"private": state["private"]})()

        def update_repo_settings(self, repo_id, private, repo_type):
            calls["settings"] = private
            if not stays_public:
                state["private"] = private

        def list_repo_refs(self, repo_id, repo_type):
            return type("Refs", (), {"tags": [
                type("Tag", (), {"name": t, "target_commit": f"sha-{t}"})() for t in tags]})()

        def create_repo(self, **k):
            calls["repo"] = k
            state["private"] = bool(k.get("private")) and not comes_out_public

        def upload_folder(self, **k):
            calls["upload"] = k
            return type("Info", (), {"oid": "abc123"})()

        def create_tag(self, **k):
            calls["tag"] = k

    monkeypatch.setattr("huggingface_hub.HfApi", Hub)
    return calls


def _stored_version(tmp_path, monkeypatch, complete: bool = True):
    from dfpipeline import lerobot_builder as B
    from dfpipeline.storage import LocalStorage, use_storage

    use_storage(LocalStorage(str(tmp_path / "store")))
    root = tmp_path / "store" / "datasets" / "d" / "e"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    if complete:
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"x")
        (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"x")
    monkeypatch.setattr(B.settings, "hf_token", "hf_test")
    monkeypatch.setattr(B, "_resolve_repo_id", lambda repo_id, token: "me/" + repo_id.split("/")[-1])
    return B


def test_publishing_replaces_the_files_and_tags_the_version(tmp_path, monkeypatch):
    B = _stored_version(tmp_path, monkeypatch)
    calls = _hub(monkeypatch)
    out = B.publish("datasets/d/e/", "someone/redmind", True, version="0.2.0")

    assert out["repo_id"] == "me/redmind" and out["tag"] == "v0.2.0"
    assert "meta/**" in calls["upload"]["delete_patterns"], "stale files of the last version go"
    assert calls["tag"]["tag"] == "v0.2.0" and calls["tag"]["revision"] == "abc123"
    assert calls["repo"]["private"] is True


def test_a_private_dataset_is_private_before_anything_is_uploaded(tmp_path, monkeypatch):
    B = _stored_version(tmp_path, monkeypatch)
    calls = _hub(monkeypatch, comes_out_public=True)
    B.publish("datasets/d/e/", "someone/redmind", True, version="0.2.0")
    assert calls["settings"] is True, "made private"
    assert "upload" in calls


def test_nothing_is_uploaded_to_a_repo_that_will_not_go_private(tmp_path, monkeypatch):
    B = _stored_version(tmp_path, monkeypatch)
    calls = _hub(monkeypatch, comes_out_public=True, stays_public=True)
    with pytest.raises(RuntimeError, match="could not be made private"):
        B.publish("datasets/d/e/", "someone/redmind", True, version="0.2.0")
    assert "upload" not in calls


def test_a_version_already_on_the_hub_is_not_pushed_again(tmp_path, monkeypatch):
    """A second push of a version runs after the first deleted the stored
    copy. Uploading what was left emptied a published repo once."""
    B = _stored_version(tmp_path, monkeypatch, complete=False)
    calls = _hub(monkeypatch, tags=("v0.1.0",))
    out = B.publish("datasets/d/e/", "someone/redmind", True, version="0.1.0")

    assert "upload" not in calls and "tag" not in calls, "the repo is left alone"
    assert out["tag"] == "v0.1.0" and out["commit_sha"] == "sha-v0.1.0"


def test_an_incomplete_copy_is_never_published(tmp_path, monkeypatch):
    B = _stored_version(tmp_path, monkeypatch, complete=False)
    calls = _hub(monkeypatch)
    with pytest.raises(RuntimeError, match="incomplete"):
        B.publish("datasets/d/e/", "someone/redmind", True, version="0.3.0")
    assert "upload" not in calls


# --- a robot dataset in LeRobot's own SO-101 shape ------------------------------


def _robot_world(monkeypatch, tmp_path, lengths: dict[str, int], frames: int = 40):
    """A robot build: each clip's trajectory has `lengths[clip]` action samples."""
    B = _world(monkeypatch, tmp_path, frames=frames)
    from dfpipeline.retarget import service as rs

    def load(path):
        n = lengths[path.rsplit("/", 1)[-1]]
        return {"trajectory": [
            {"phase": "action", "frame": i, "timestamp": i / 30,
             "action": {f"{j}.pos": float(i) for j in B.SO101_JOINTS}}
            for i in range(n)
        ]}

    monkeypatch.setattr(rs, "load", load)
    return B


def _robot_episode(name: str, steps=None) -> dict:
    ep = {"task": "t", "clip": _clip(name), "video_id": name, "trajectory_path": f"r/{name}"}
    if steps:
        ep["subtasks"] = steps
    return ep


def _robot_payload(name: str, episodes, **extra) -> dict:
    return _payload(name, episodes, robot_id="r1", robot_kind="so101", **extra)


def test_a_robot_dataset_says_it_is_an_so101_follower(tmp_path, monkeypatch):
    """The visualizer's 3D replay poses the arm only for a known robot_type and
    LeRobot's own joint names; without them it says "robot type unknown"."""
    B = _robot_world(monkeypatch, tmp_path, {"a": 40})
    result = B.build_to_store(_robot_payload("arm", [_robot_episode("a")]), tmp_path / "ds-arm")

    info = json.loads((Path(result["root"]) / "meta" / "info.json").read_text())
    assert info["robot_type"] == "so101_follower"
    names = [f"{j}.pos" for j in B.SO101_JOINTS]
    assert info["features"]["action"]["names"] == names
    assert info["features"]["observation.state"]["names"] == names


def test_an_episode_too_short_to_train_on_is_left_out(tmp_path, monkeypatch):
    B = _robot_world(monkeypatch, tmp_path, {"long": 40, "short": 3})
    monkeypatch.setattr(B.settings, "export_min_episode_frames", 30)
    result = B.build_to_store(
        _robot_payload("short", [_robot_episode("long"), _robot_episode("short")]),
        tmp_path / "ds-short",
    )
    assert result["episodes"] == 1 and result["video_ids"] == ["long"]


def test_an_older_version_is_brought_up_to_date_when_merged(tmp_path, monkeypatch):
    """A version built before these rules has no robot_type, bare joint names
    and a one-frame episode. Merged onto, it gains the first two and loses the
    third, and keeps its subtasks."""
    import pandas as pd

    B = _robot_world(monkeypatch, tmp_path, {"tiny": 1, "a": 40, "b": 40})
    steps = [{"name": "reach", "start": 0.0, "end": 0.5}, {"name": "grasp", "start": 0.5, "end": 1.3}]
    B.build_to_store(
        _robot_payload("old", [_robot_episode("tiny"), _robot_episode("a", steps)],
                       store_prefix="datasets/old/v1"),
        tmp_path / "ds-v1",
    )
    # As the old builder wrote it.
    old_info = tmp_path / "store" / "datasets" / "old" / "v1" / "meta" / "info.json"
    info = json.loads(old_info.read_text())
    info["robot_type"] = None
    for key in ("action", "observation.state"):
        info["features"][key]["names"] = list(B.SO101_JOINTS)
    old_info.write_text(json.dumps(info))

    monkeypatch.setattr(B.settings, "export_min_episode_frames", 30)
    v2 = B.build_to_store(
        _robot_payload("old", [_robot_episode("b", steps)], store_prefix="datasets/old/v2",
                       base={"store_prefix": "datasets/old/v1/"}),
        tmp_path / "ds-v2",
    )

    stored = tmp_path / "store" / "datasets" / "old" / "v2"
    info = json.loads((stored / "meta" / "info.json").read_text())
    assert info["robot_type"] == "so101_follower"
    assert info["features"]["action"]["names"][0] == "shoulder_pan.pos"
    assert v2["episodes"] == 2, "the one-frame episode is gone; a and b remain"
    episodes = pd.concat(pd.read_parquet(p) for p in (stored / "meta" / "episodes").rglob("*.parquet"))
    assert episodes["length"].min() >= 30
    names = episodes.sort_values("episode_index")["sparse_subtask_names"].map(list).tolist()
    assert names == [["reach", "grasp"], ["reach", "grasp"]], "subtasks survive the trim"


def test_the_hold_comes_from_the_steps_the_check_named():
    from dfpipeline.lerobot_builder import _hold_window

    steps = [
        {"name": "reach for the cup", "start": 0.0, "end": 5.4},
        {"name": "pick up the cup", "start": 5.4, "end": 8.9},
        {"name": "hold the cup", "start": 8.9, "end": 11.8},
        {"name": "place the cup back", "start": 11.8, "end": 13.9},
        {"name": "withdraw the hand", "start": 13.9, "end": 15.0},
    ]
    # Closed as "pick up" begins; open half way through "place", as it lands.
    assert _hold_window(steps) == (5.4, (11.8 + 13.9) / 2)
    assert _hold_window([{"name": "wave at the camera", "start": 0.0, "end": 2.0}]) is None
    assert _hold_window([]) is None


def test_the_gripper_is_closed_while_the_object_is_held(tmp_path, monkeypatch):
    """A hand around a cup keeps thumb and index apart, so the pinch never
    closed the gripper. Over the held steps it is closed."""
    import numpy as np
    import pandas as pd

    B = _robot_world(monkeypatch, tmp_path, {"a": 40})
    steps = [{"name": "reach", "start": 0.0, "end": 0.5},
             {"name": "pick up the cup", "start": 0.5, "end": 0.9},
             {"name": "put it down", "start": 0.9, "end": 1.3}]
    result = B.build_to_store(_robot_payload("held", [_robot_episode("a", steps)]), tmp_path / "ds-held")

    data = pd.read_parquet(next((Path(result["root"]) / "data").rglob("*.parquet")))
    grip = np.stack(data["action"].to_numpy())[:, -1]
    t = data["frame_index"].to_numpy() / 30
    assert (grip[(t >= 0.5) & (t <= 1.1)] == 0.0).all(), "closed from the pick until it lands"
    assert (grip[t < 0.5] == B.GRIPPER_OPEN).all(), "open while reaching, whatever the fingers did"
    assert (grip[t > 1.1] == B.GRIPPER_OPEN).all(), "open once it is set down"


# --- a version on the Hub is added to with new files only ------------------------


def _fake_hub_download(monkeypatch, hub: Path, fetched: list):
    """snapshot_download from a folder standing in for the Hub; records what
    was fetched, so a test can say no old data or video came down."""
    import fnmatch
    import shutil as sh

    def snapshot_download(repo_id, repo_type, revision, local_dir, token=None, allow_patterns=None):
        for path in hub.rglob("*"):
            rel = path.relative_to(hub).as_posix()
            if path.is_file() and (allow_patterns is None or any(fnmatch.fnmatch(rel, p) for p in allow_patterns)):
                (Path(local_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
                sh.copy2(path, Path(local_dir) / rel)
                fetched.append(rel)
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)


def _push(stored: Path, hub: Path):
    """What publishing a version of new files does to the repo: they are added
    beside the old ones, nothing is deleted."""
    import shutil as sh

    for path in stored.rglob("*"):
        if path.is_file() and not path.name.startswith(".export-"):
            (hub / path.relative_to(stored)).parent.mkdir(parents=True, exist_ok=True)
            sh.copy2(path, hub / path.relative_to(stored))


def _files(root: Path, part: str) -> set[str]:
    return {p.relative_to(root).as_posix() for p in (root / part).rglob("*") if p.is_file()}


def test_a_version_on_the_hub_is_added_to_with_new_files_only(tmp_path, monkeypatch):
    """v1 is on the Hub. v2 fetches its index alone — no old data, no old
    video — writes the new episodes into new files, and published beside v1's
    files the repo is one dataset LeRobot reads whole."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    B = _robot_world(monkeypatch, tmp_path, {"a": 40, "b": 40, "c": 40, "d": 40})
    steps = [{"name": "reach", "start": 0.0, "end": 0.5}, {"name": "grasp", "start": 0.5, "end": 1.2}]
    B.build_to_store(
        _robot_payload("grow", [_robot_episode("a", steps), _robot_episode("b")],
                       store_prefix="datasets/grow/v1"),
        tmp_path / "ds-v1",
    )
    hub = tmp_path / "hub"
    _push(tmp_path / "store" / "datasets" / "grow" / "v1", hub)
    old_data, old_videos = _files(hub, "data"), _files(hub, "videos")

    fetched: list[str] = []
    _fake_hub_download(monkeypatch, hub, fetched)
    v2 = B.build_to_store(
        _robot_payload("grow", [_robot_episode("c", steps), _robot_episode("d")],
                       store_prefix="datasets/grow/v2",
                       base={"repo_id": "me/grow", "revision": "v0.1.0"}),
        tmp_path / "ds-v2",
    )

    assert fetched and all(f.startswith("meta/") for f in fetched), "only the index came down"
    stored = tmp_path / "store" / "datasets" / "grow" / "v2"
    assert not (_files(stored, "data") & old_data), "no old data file is in the new version"
    assert not (_files(stored, "videos") & old_videos), "no old video is in the new version"
    assert _files(stored, "data") and _files(stored, "videos")
    assert json.loads((stored / ".export-append.json").read_text())["revision"] == "v0.1.0"
    assert v2["episodes"] == 4 and v2["video_ids"] == ["c", "d"]

    _push(stored, hub)
    ds = LeRobotDataset("me/grow", root=hub)
    assert ds.meta.total_episodes == 4 and ds.num_frames == 160
    assert [int(i) for i in ds.hf_dataset["index"]] == list(range(160)), "frame numbers run on without a gap"
    assert sorted({int(e) for e in ds.hf_dataset["episode_index"]}) == [0, 1, 2, 3]
    last = ds[159]
    assert int(last["episode_index"]) == 3 and "observation.images.top" in last
    episodes = ds.meta.episodes.to_pandas().sort_values("episode_index")
    names = episodes["subtask_names"].map(lambda v: list(v) if v is not None else None).tolist()
    assert names == [["reach", "grasp"], None, ["reach", "grasp"], None], "timelines on old and new"


def test_versions_not_pushed_yet_carry_their_new_files_forward(tmp_path, monkeypatch):
    """v2 (new files only) is built but not pushed when v3 is built on it: v3
    holds v2's new files too, so publishing v3 alone leaves nothing out."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    B = _robot_world(monkeypatch, tmp_path, {"a": 40, "b": 40, "c": 40})
    B.build_to_store(_robot_payload("chain", [_robot_episode("a")], store_prefix="datasets/chain/v1"),
                     tmp_path / "ds-v1")
    hub = tmp_path / "hub"
    _push(tmp_path / "store" / "datasets" / "chain" / "v1", hub)
    _fake_hub_download(monkeypatch, hub, [])
    B.build_to_store(_robot_payload("chain", [_robot_episode("b")], store_prefix="datasets/chain/v2",
                                    base={"repo_id": "me/chain", "revision": "v0.1.0"}),
                     tmp_path / "ds-v2")
    v3 = B.build_to_store(_robot_payload("chain", [_robot_episode("c")], store_prefix="datasets/chain/v3",
                                         base={"store_prefix": "datasets/chain/v2/"}),
                          tmp_path / "ds-v3")

    stored = tmp_path / "store" / "datasets" / "chain" / "v3"
    assert json.loads((stored / ".export-append.json").read_text())["revision"] == "v0.1.0"
    assert v3["episodes"] == 3
    _push(stored, hub)
    ds = LeRobotDataset("me/chain", root=hub)
    assert ds.meta.total_episodes == 3 and ds.num_frames == 120


def test_a_layout_change_is_merged_in_full(tmp_path, monkeypatch):
    """New episodes that do not fit the old files (here, another robot) are not
    put beside them: the previous version is fetched whole and merged."""
    B = _robot_world(monkeypatch, tmp_path, {"a": 40, "b": 40})
    B.build_to_store(_robot_payload("swap", [_robot_episode("a")], store_prefix="datasets/swap/v1"),
                     tmp_path / "ds-v1")
    hub = tmp_path / "hub"
    _push(tmp_path / "store" / "datasets" / "swap" / "v1", hub)
    fetched: list[str] = []
    _fake_hub_download(monkeypatch, hub, fetched)
    info = json.loads((hub / "meta" / "info.json").read_text())
    info["robot_type"] = "koch_follower"
    (hub / "meta" / "info.json").write_text(json.dumps(info))

    v2 = B.build_to_store(_robot_payload("swap", [_robot_episode("b")], store_prefix="datasets/swap/v2",
                                         base={"repo_id": "me/swap", "revision": "v0.1.0"}),
                          tmp_path / "ds-v2")
    stored = tmp_path / "store" / "datasets" / "swap" / "v2"
    assert not (stored / ".export-append.json").exists(), "a whole dataset"
    assert any(f.startswith("videos/") for f in fetched), "fetched whole to be merged"
    assert v2["episodes"] == 2


def test_publishing_new_files_deletes_nothing_on_the_hub(tmp_path, monkeypatch):
    B = _stored_version(tmp_path, monkeypatch)
    root = tmp_path / "store" / "datasets" / "d" / "e"
    (root / ".export-append.json").write_text(json.dumps({"repo_id": "me/redmind", "revision": "v0.1.0"}))
    calls = _hub(monkeypatch, tags=("v0.1.0",))
    B.publish("datasets/d/e/", "someone/redmind", True, version="0.2.0")
    assert "delete_patterns" not in calls["upload"], "the old episodes stay"
    assert calls["upload"]["ignore_patterns"] == [".export-*"]
    assert calls["tag"]["tag"] == "v0.2.0"


def test_new_files_are_not_published_onto_a_repo_without_their_version(tmp_path, monkeypatch):
    B = _stored_version(tmp_path, monkeypatch)
    root = tmp_path / "store" / "datasets" / "d" / "e"
    (root / ".export-append.json").write_text(json.dumps({"repo_id": "me/redmind", "revision": "v0.1.0"}))
    calls = _hub(monkeypatch)
    with pytest.raises(RuntimeError, match="publish that one first"):
        B.publish("datasets/d/e/", "someone/redmind", True, version="0.2.0")
    assert "upload" not in calls


def test_a_build_never_deletes_a_clip_it_did_not_fetch(tmp_path, monkeypatch):
    """A clip read where it lies — a local file, or the development store — is
    the original: the builder deletes only the copies it fetched. It once
    deleted a video straight out of someone's Downloads."""
    B = _world(monkeypatch, tmp_path)
    original = tmp_path / "Downloads" / "clip.mp4"
    original.parent.mkdir()
    original.write_bytes(b"the only copy")
    monkeypatch.setattr(B.sources, "materialize", lambda spec, work_dir=None: original)

    B.build_to_store(_payload("keep", [{"task": "t", "clip": _clip("a")}, {"task": "t", "clip": _clip("b")}]),
                     tmp_path / "ds")
    assert original.read_bytes() == b"the only copy"


def test_a_fetched_copy_is_still_cleaned_up(tmp_path):
    import tempfile

    from dfpipeline.lerobot_builder import _fetched_copy

    work = tmp_path / "work"
    work.mkdir()
    assert _fetched_copy(work / "clip.mp4", work)
    assert _fetched_copy(Path(tempfile.gettempdir()) / "dataforge-fetch" / "raw" / "c.mp4", work)
    assert not _fetched_copy(tmp_path / "Downloads" / "clip.mp4", work)
