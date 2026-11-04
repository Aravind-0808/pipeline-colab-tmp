"""Reading a clip described by the app server, from the store (MinIO).

The app describes each clip by its key in the bucket (CONTRACT.md); this reads
it with the pipeline's own access key — never a credential carried in a job.

An app that keeps its clips in a folder (development) describes a clip as
`kind: "local"` with the file's absolute path on its own disk. That path is
only good on the same machine. A worker elsewhere — Colab, reaching the
laptop's folder through S3 (run-web-for-colab.sh) — reads the same clip by its
key (`path`, always given) from its store instead.
"""

from __future__ import annotations

from pathlib import Path

from dfpipeline.storage import S3Storage, get_storage


def _on_this_disk(spec: dict) -> bool:
    """A "local" clip this worker can open where the app said it is."""
    return spec.get("kind") == "local" and Path(spec["local"]).is_file()


def _from_store(spec: dict) -> bool:
    """Read by key from the store: a clip in MinIO, or a "local" one whose
    path is on another machine's disk while this worker has a store of its own."""
    if spec.get("kind") != "local":
        return True
    return not _on_this_disk(spec) and isinstance(get_storage(), S3Storage) and bool(spec.get("path"))


def resolve(spec: dict) -> tuple[str, dict | None]:
    """(a path or URL ffmpeg can open, headers to send with it)."""
    if not _from_store(spec):
        return spec["local"], None
    storage = get_storage()
    if isinstance(storage, S3Storage):
        # A short-lived link on MinIO's own address: ffmpeg reads only the
        # frames it needs from it, rather than the whole clip.
        return storage.presigned(spec["path"], 60), None
    return str(storage.fetch_local(spec["path"])), None


def materialize(spec: dict, work_dir: Path | None = None) -> Path:
    """The clip as a file on this machine's disk, downloading it if need be."""
    if not _from_store(spec):
        return Path(spec["local"])
    return get_storage().fetch_local(spec["path"])
