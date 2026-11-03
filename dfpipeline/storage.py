"""Where clips and everything made from them are kept: MinIO (S3).

The app server's bucket (S3_BUCKET), reached over the private network with the
pipeline's own access key. Clips are read from it; what the pipeline makes —
hand tracks, datasets — is written to it. Hugging Face only ever gets a
finished dataset, when an admin publishes it.

A copy of the app's app/core/storage.py, less what only the app does (the 20 GB
admission check, signed media links): keep the two the same.

Paths inside the bucket:

    <project>/<task>/<clip>.<ext>       uploaded clips
    tracks/<version>/<clip>.zip          hand tracking (written by the pipeline)
    tasks/<task>/reference.<ext>         a task's reference image
    datasets/<name>/v<N>/...             built datasets (written by the pipeline)

Without S3_ENDPOINT — development and tests — files are kept in a folder
(STORAGE_ROOT) instead, through the same interface.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO

from dfpipeline.config import settings

log = logging.getLogger(__name__)


class StorageUnavailable(Exception):
    """The store refused or could not take the file. Trying later can work."""


class StoredFile:
    def __init__(self, path: str, size_bytes: int, checksum: str):
        self.path = path
        self.size_bytes = size_bytes
        self.checksum = checksum


class Storage(ABC):
    @abstractmethod
    def put(self, path: str, stream: BinaryIO) -> StoredFile: ...

    @abstractmethod
    def delete(self, path: str) -> None: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def fetch_local(self, path: str) -> Path:
        """A readable local path (downloaded first when the store is remote)."""

    @abstractmethod
    def used_bytes(self) -> int:
        """How much the store holds — what the 20 GB limit is checked against."""


class _Hashing:
    """A stream that counts and hashes what is read through it."""

    def __init__(self, stream: BinaryIO):
        self.stream, self.sha, self.size = stream, hashlib.sha256(), 0

    def read(self, n: int = -1) -> bytes:
        chunk = self.stream.read(n)
        self.sha.update(chunk)
        self.size += len(chunk)
        return chunk


class LocalStorage(Storage):
    """A folder — development and tests, where there is no MinIO."""

    def __init__(self, root: str):
        self.root = Path(root)

    def _abs(self, path: str) -> Path:
        return self.root / path

    def put(self, path: str, stream: BinaryIO) -> StoredFile:
        dest = self._abs(path)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            reader = _Hashing(stream)
            with dest.open("wb") as out:
                while chunk := reader.read(1 << 20):
                    out.write(chunk)
        except OSError as exc:
            raise StorageUnavailable(f"could not store {path}: {exc}") from exc
        return StoredFile(path=path, size_bytes=reader.size, checksum=reader.sha.hexdigest())

    def delete(self, path: str) -> None:
        self._abs(path).unlink(missing_ok=True)

    def exists(self, path: str) -> bool:
        return self._abs(path).exists()

    def fetch_local(self, path: str) -> Path:
        local = self._abs(path)
        if not local.is_file():
            raise FileNotFoundError(path)
        return local

    def upload_dir(self, local: Path, prefix: str) -> int:
        """Every file under `local`, to `prefix`. Returns how many."""
        files = [f for f in Path(local).rglob("*") if f.is_file()]
        for f in files:
            dest = self._abs(prefix + f.relative_to(local).as_posix())
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(f.read_bytes())
        return len(files)

    def download_prefix(self, prefix: str, dest: Path) -> int:
        """Every file under `prefix`, into `dest`. Returns how many."""
        src = self._abs(prefix)
        files = [f for f in src.rglob("*") if f.is_file()] if src.is_dir() else []
        for f in files:
            out = Path(dest) / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(f.read_bytes())
        return len(files)

    def delete_prefix(self, prefix: str) -> int:
        import shutil

        src = self._abs(prefix)
        count = sum(1 for f in src.rglob("*") if f.is_file()) if src.is_dir() else 0
        shutil.rmtree(src, ignore_errors=True)
        return count

    def used_bytes(self) -> int:
        if not self.root.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


#: Where files fetched from MinIO are put on this machine.
FETCH_DIR = "dataforge-fetch"
#: Fetched once and kept: named by their content's hash, so a kept copy is
#: never stale (a changed robot is a new name).
KEPT_PREFIXES = ("robots/packages/",)
#: A kept copy unused this long is removed.
KEEP_UNUSED_SEC = 7 * 86400
#: Any other fetched file (a clip, a hand track, a trajectory) is a copy for
#: the job that asked: removed once it is this old. Nothing removed them, and
#: the pipeline server's disk filled with every clip it had ever checked.
FETCHED_MAX_AGE_SEC = 86400
#: Looked over once a week. When is kept on disk (PRUNE_MARKER's time), so
#: a worker restarting does not look over it again early.
PRUNE_EVERY_SEC = 7 * 86400
PRUNE_MARKER = ".pruned"


@contextlib.contextmanager
def _file_lock(dest: Path):
    """Held by one process at a time, for `dest` (a lock file beside it): the
    pipeline's workers are separate processes, so a lock in memory is not
    enough. Where there is no fcntl (Windows, in development), no lock."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(dest.with_name(f"{dest.name}.lock"), "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _prune_fetched(root: Path) -> None:
    """Remove fetched copies past their age, and downloads a crash left
    half-written. Never a file in use: anything this old has been finished
    with, and a kept package is touched each time it is used."""
    now = time.time()
    marker = root / PRUNE_MARKER
    try:
        if now - marker.stat().st_mtime < PRUNE_EVERY_SEC:
            return
    except FileNotFoundError:
        pass  # never looked over here: the first fetch starts the week
    marker.touch()
    kept = tuple(root / k for k in KEPT_PREFIXES)
    for f in root.rglob("*"):
        try:
            if not f.is_file() or f == marker:
                continue
            in_kept = any(f.is_relative_to(k) for k in kept)
            scratch = f.name.endswith((".part", ".lock"))
            limit = KEEP_UNUSED_SEC if in_kept and not scratch else FETCHED_MAX_AGE_SEC
            if now - f.stat().st_mtime > limit:
                f.unlink(missing_ok=True)
        except (OSError, ValueError):
            continue  # gone already: the next look tries again


class S3Storage(Storage):
    """A MinIO (or any S3) bucket."""

    # Read from MinIO at most this often: listing the bucket is the only way to
    # know its size, and every upload asks.
    USED_CACHE_SEC = 30

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        from urllib.parse import urlparse

        from minio import Minio

        url = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
        self.bucket = bucket
        self.client = Minio(
            url.netloc, access_key=access_key, secret_key=secret_key, secure=url.scheme == "https"
        )
        self._used: tuple[float, int] | None = None
        self._lock = threading.Lock()
        self._ready = False

    def ensure_bucket(self) -> None:
        if self._ready:
            return
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
        self._ready = True

    def put(self, path: str, stream: BinaryIO) -> StoredFile:
        reader = _Hashing(stream)
        try:
            self.ensure_bucket()
            self.client.put_object(self.bucket, path, reader, length=-1, part_size=10 * 1024 * 1024)
        except Exception as exc:
            # The bucket over its quota, MinIO restarting: nothing is wrong
            # with the file, so this is "try later", never "record again".
            raise StorageUnavailable(f"could not store {path}: {exc}") from exc
        with self._lock:
            if self._used:
                self._used = (self._used[0], self._used[1] + reader.size)
        return StoredFile(path=path, size_bytes=reader.size, checksum=reader.sha.hexdigest())

    def delete(self, path: str) -> None:
        try:
            self.client.remove_object(self.bucket, path)
        except Exception:
            # Logged, not raised: a delete that fails must not fail the request.
            log.warning("could not delete %s from %s", path, self.bucket, exc_info=True)

    def delete_prefix(self, prefix: str) -> int:
        """Delete everything under `prefix` (a dataset version). Returns how many."""
        from minio.deleteobjects import DeleteObject

        names = [o.object_name for o in self.client.list_objects(self.bucket, prefix, recursive=True)]
        errors = list(self.client.remove_objects(self.bucket, (DeleteObject(n) for n in names)))
        for err in errors:
            log.warning("could not delete %s: %s", err.name, err.message)
        return len(names) - len(errors)

    def upload_dir(self, local: Path, prefix: str) -> int:
        """Every file under `local`, to `prefix` (a dataset version). Returns how many."""
        self.ensure_bucket()
        files = [f for f in Path(local).rglob("*") if f.is_file()]
        for f in files:
            self.client.fput_object(self.bucket, prefix + f.relative_to(local).as_posix(), str(f))
        return len(files)

    def download_prefix(self, prefix: str, dest: Path) -> int:
        """Every file under `prefix`, into `dest`. Returns how many."""
        count = 0
        for obj in self.client.list_objects(self.bucket, prefix, recursive=True):
            out = Path(dest) / obj.object_name[len(prefix):]
            out.parent.mkdir(parents=True, exist_ok=True)
            self.client.fget_object(self.bucket, obj.object_name, str(out))
            count += 1
        return count

    def exists(self, path: str) -> bool:
        from minio.error import S3Error

        try:
            self.client.stat_object(self.bucket, path)
            return True
        except S3Error:
            return False

    def fetch_local(self, path: str) -> Path:
        root = Path(tempfile.gettempdir()) / FETCH_DIR
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        _prune_fetched(root)
        if path.startswith(KEPT_PREFIXES):
            # A robot's package: fetched once, then reused. It was downloaded
            # for every clip — 15 clips, 15 downloads of the same zip. Its name
            # is its content's hash, so a copy here is never out of date; MinIO
            # is still asked that it exists (a few hundred bytes), so a robot
            # deleted in the app is never used from a copy left here.
            try:
                info = self.client.stat_object(self.bucket, path)
            except Exception as exc:
                if getattr(exc, "code", None) in ("NoSuchKey", "NoSuchBucket"):
                    dest.unlink(missing_ok=True)
                    raise FileNotFoundError(path) from exc
                raise
            # One download however many jobs ask at once: the first takes
            # the lock and fetches, the rest wait and reuse its copy.
            with _file_lock(dest):
                if dest.is_file() and dest.stat().st_size == info.size:
                    os.utime(dest)  # in use: kept another KEEP_UNUSED_SEC
                    return dest
                return self._download(path, dest)
        return self._download(path, dest)

    def _download(self, path: str, dest: Path) -> Path:
        # Each download to a file of its own, then moved into place at once.
        # Downloaded straight to `dest`, two jobs fetching the same file (two
        # clips solved on one robot, its package) shared MinIO's temporary
        # name: one renamed it from under the other (FileNotFoundError ...
        # .part.minio) or read a zip still being written (BadZipFile).
        part = dest.with_name(f"{dest.name}.{uuid.uuid4().hex}.part")
        try:
            self.client.fget_object(self.bucket, path, str(part))
            os.replace(part, dest)
        except Exception as exc:
            part.unlink(missing_ok=True)
            if getattr(exc, "code", None) in ("NoSuchKey", "NoSuchBucket"):
                raise FileNotFoundError(path) from exc
            raise
        return dest

    def presigned(self, path: str, minutes: int = 60) -> str:
        """A short-lived link to read `path`, on this server's MinIO address."""
        return self.client.presigned_get_object(self.bucket, path, expires=timedelta(minutes=minutes))

    def used_bytes(self) -> int:
        with self._lock:
            if self._used and time.monotonic() - self._used[0] < self.USED_CACHE_SEC:
                return self._used[1]
        try:
            total = sum(o.size or 0 for o in self.client.list_objects(self.bucket, recursive=True))
        except Exception:
            log.warning("could not measure %s", self.bucket, exc_info=True)
            return self._used[1] if self._used else 0
        with self._lock:
            self._used = (time.monotonic(), total)
        return total


# `_override` is a test hook: the suite injects a LocalStorage of its own.
_override: Storage | None = None
_cached: Storage | None = None


def use_storage(storage: Storage | None) -> None:
    global _override, _cached
    _override = storage
    _cached = None


def get_storage() -> Storage:
    """The one store: MinIO when S3_ENDPOINT is set, a folder otherwise."""
    global _cached
    if _override is not None:
        return _override
    if _cached is None:
        if settings.s3_endpoint:
            _cached = S3Storage(
                settings.s3_endpoint, settings.s3_access_key, settings.s3_secret_key, settings.s3_bucket
            )
        else:
            _cached = LocalStorage(settings.storage_root)
    return _cached
