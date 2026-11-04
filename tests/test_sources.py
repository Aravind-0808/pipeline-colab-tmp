"""A clip the app describes by a path on its own disk, read by a worker on
another machine (Colab, reaching the laptop's folder through S3)."""

from pathlib import Path

import pytest

from dfpipeline import sources, storage


class _FakeS3(storage.S3Storage):
    def __init__(self, root: Path):   # no MinIO: files from a folder, by key
        self.root, self.asked = root, []

    def presigned(self, path, seconds):
        self.asked.append(path)
        return f"http://store/{path}"

    def fetch_local(self, path):
        self.asked.append(path)
        return self.root / path


@pytest.fixture
def s3(tmp_path):
    fake = _FakeS3(tmp_path)
    storage.use_storage(fake)
    yield fake
    storage.use_storage(None)


def _spec(local: Path) -> dict:
    return {"video_id": "v1", "path": "task/v1.mp4", "kind": "local", "local": str(local)}


def test_a_path_on_this_disk_is_read_there(tmp_path, s3):
    clip = tmp_path / "here.mp4"
    clip.write_bytes(b"x")
    assert sources.resolve(_spec(clip)) == (str(clip), None)
    assert sources.materialize(_spec(clip)) == clip
    assert s3.asked == []


def test_a_path_on_another_machine_is_read_by_key_from_the_store(tmp_path, s3):
    elsewhere = Path("/home/someone/storage/task/v1.mp4")   # the laptop's path, not here
    assert sources.resolve(_spec(elsewhere)) == ("http://store/task/v1.mp4", None)
    assert sources.materialize(_spec(elsewhere)) == tmp_path / "task/v1.mp4"
    assert s3.asked == ["task/v1.mp4", "task/v1.mp4"]


def test_without_a_store_of_its_own_the_path_is_used_as_given(tmp_path):
    storage.use_storage(storage.LocalStorage(str(tmp_path)))
    try:
        missing = Path("/nowhere/v1.mp4")
        assert sources.resolve(_spec(missing)) == (str(missing), None)   # fails later, as before
    finally:
        storage.use_storage(None)
