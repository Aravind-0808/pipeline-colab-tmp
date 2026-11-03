"""Reading a clip described by the app server, from the store (MinIO).

The app describes each clip by its key in the bucket (CONTRACT.md); this reads
it with the pipeline's own access key — never a credential carried in a job.
"""

from __future__ import annotations

from pathlib import Path

from dfpipeline.storage import S3Storage, get_storage


def resolve(spec: dict) -> tuple[str, dict | None]:
    """(a path or URL ffmpeg can open, headers to send with it)."""
    if spec.get("kind") == "local":
        return spec["local"], None
    storage = get_storage()
    if isinstance(storage, S3Storage):
        # A short-lived link on MinIO's own address: ffmpeg reads only the
        # frames it needs from it, rather than the whole clip.
        return storage.presigned(spec["path"], 60), None
    return str(storage.fetch_local(spec["path"])), None


def materialize(spec: dict, work_dir: Path | None = None) -> Path:
    """The clip as a file on this machine's disk, downloading it if need be."""
    if spec.get("kind") == "local":
        return Path(spec["local"])
    return get_storage().fetch_local(spec["path"])
