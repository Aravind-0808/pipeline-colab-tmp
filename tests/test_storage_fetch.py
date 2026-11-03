"""Fetching from MinIO onto this machine: safely when jobs fetch at once, a
robot's package once rather than for every clip, and the disk kept tidy."""

import os
import threading
import time
import zipfile
from types import SimpleNamespace

import pytest

from dfpipeline import storage as st

PACKAGE = "robots/packages/abc123.zip"
TRACK = "tracks/v/clip.zip"


class _Missing(Exception):
    code = "NoSuchKey"


class FakeMinio:
    """Writes downloads slowly, a chunk at a time, through MinIO's own fixed
    temporary name, as the real client does — so two jobs fetching one file
    overlap the way they did on the pipeline server."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.downloads = 0
        self.lock = threading.Lock()

    def stat_object(self, bucket, path):
        if path not in self.objects:
            raise _Missing()
        return SimpleNamespace(size=len(self.objects[path]))

    def fget_object(self, bucket, path, file_path):
        if path not in self.objects:
            raise _Missing()
        with self.lock:
            self.downloads += 1
        data = self.objects[path]
        tmp = f"{file_path}.etag.part.minio"
        with open(tmp, "wb") as f:
            for i in range(0, len(data), 4096):
                f.write(data[i:i + 4096])
                f.flush()
                time.sleep(0.0005)
        os.rename(tmp, file_path)


def _zip_bytes(tmp_path, name) -> bytes:
    z = tmp_path / name
    with zipfile.ZipFile(z, "w") as out:
        out.writestr("robot.urdf", "<robot name='r'/>" * 2000)
        out.writestr("meshes/link.stl", os.urandom(80_000))
    return z.read_bytes()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(st.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    s = object.__new__(st.S3Storage)
    s.bucket = "b"
    s.client = FakeMinio({PACKAGE: _zip_bytes(tmp_path, "p.zip"), TRACK: _zip_bytes(tmp_path, "t.zip")})
    return s


@pytest.mark.parametrize("path", [TRACK, PACKAGE])
def test_jobs_fetching_one_file_at_once_all_get_it_whole(store, path):
    """Two jobs downloading one robot's zip shared MinIO's temporary name:
    FileNotFoundError ... .part.minio, or BadZipFile from a half-written zip."""
    errors = []

    def fetch():
        try:
            with zipfile.ZipFile(store.fetch_local(path)) as z:
                assert z.testzip() is None
        except Exception as exc:  # noqa: BLE001 - the test reports any
            errors.append(repr(exc))

    threads = [threading.Thread(target=fetch) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    folder = (store.fetch_local(path)).parent
    assert not [f for f in folder.iterdir() if "part" in f.name], "no half-written file is left"


def test_a_robots_package_is_fetched_once_and_reused(store):
    """15 clips on one robot downloaded its zip 15 times."""
    for _ in range(15):
        p = store.fetch_local(PACKAGE)
    assert store.client.downloads == 1
    with zipfile.ZipFile(p) as z:
        assert "robot.urdf" in z.namelist()


def test_other_files_are_fetched_fresh_each_time(store):
    """A clip's track or trajectory can be written again: never reused."""
    store.fetch_local(TRACK)
    store.fetch_local(TRACK)
    assert store.client.downloads == 2


def test_a_robot_deleted_in_the_app_is_never_used_from_a_kept_copy(store):
    kept = store.fetch_local(PACKAGE)
    del store.client.objects[PACKAGE]
    with pytest.raises(FileNotFoundError):
        store.fetch_local(PACKAGE)
    assert not kept.exists(), "the copy goes with it"


def test_old_copies_are_removed_and_the_one_in_use_kept(store, monkeypatch):
    """Nothing removed fetched clips and tracks: the server's disk filled."""
    kept = store.fetch_local(PACKAGE)
    track = store.fetch_local(TRACK)
    root = kept.parents[2]
    old_package = kept.parent / "old-robot.zip"
    old_package.write_bytes(b"x")
    stale_part = kept.parent / "abc123.zip.deadbeef.part"
    stale_part.write_bytes(b"x")
    day_old = time.time() - st.FETCHED_MAX_AGE_SEC - 60
    week_old = time.time() - st.KEEP_UNUSED_SEC - 60
    os.utime(track, (day_old, day_old))
    os.utime(stale_part, (day_old, day_old))
    os.utime(old_package, (week_old, week_old))
    recent_package = kept.parent / "recent-robot.zip"
    recent_package.write_bytes(b"x")
    os.utime(recent_package, (day_old, day_old))  # a day unused: a robot is kept a week

    marker = root / st.PRUNE_MARKER
    week_ago = time.time() - st.PRUNE_EVERY_SEC - 60
    os.utime(marker, (week_ago, week_ago))  # the last look over was a week ago
    store.fetch_local(PACKAGE)
    assert not track.exists() and not stale_part.exists() and not old_package.exists()
    assert kept.exists() and recent_package.exists()
    assert root.name == st.FETCH_DIR


def test_the_disk_is_looked_over_once_a_week_not_each_fetch(store):
    """Each look walks the folder: once a week is enough, and a restart does
    not bring it forward (the time is kept on disk)."""
    track = store.fetch_local(TRACK)
    old = time.time() - st.FETCHED_MAX_AGE_SEC - 60
    os.utime(track, (old, old))
    store.fetch_local(PACKAGE)  # looked over a moment ago: not yet again
    assert track.exists()


def test_jobs_asking_for_a_package_at_once_download_it_once(store):
    """The first job takes the lock and downloads; the rest wait and reuse it."""
    threads = [threading.Thread(target=store.fetch_local, args=(PACKAGE,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.client.downloads == 1
