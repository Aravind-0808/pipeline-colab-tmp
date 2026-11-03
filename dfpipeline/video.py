import contextlib
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx



def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def clip_for_model(source: str, width: int, fps: float, max_frames: int,
                   duration: float | None, headers: dict | None = None) -> bytes:
    """The clip as a small MJPEG video, for the model to watch.

    The model server reads MJPEG but not H.264 (what phones record), so this
    re-encodes: `width` px wide, `fps` frames a second — fewer for a long clip,
    so no clip is more than `max_frames` frames — and no sound. `source` is a
    path or a link ffmpeg can open. Returns the bytes; nothing is kept.
    """
    if duration and duration > 0:
        fps = min(fps, max_frames / duration)
    fps = max(fps, 0.5)
    args = [ffmpeg_exe(), "-y", "-loglevel", "error"]
    if headers:
        args += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    args += ["-i", source, "-vf", f"fps={fps:g},scale={width}:-2", "-an",
             "-c:v", "mjpeg", "-q:v", "6", "-f", "avi", "pipe:1"]
    out = subprocess.run(args, capture_output=True, timeout=180, check=False)
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(f"could not prepare the clip for the model: {out.stderr.decode(errors='replace')[-300:]}")
    return out.stdout


def download(url: str, dest_dir: Path) -> Path:
    dest = dest_dir / f"{uuid.uuid4().hex}.mp4"
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest.open("wb") as out:
            for chunk in resp.iter_bytes(1 << 20):
                out.write(chunk)
    return dest


def probe_size(path: Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return h, w
    finally:
        cap.release()


def frame_filter(fps: int, size: tuple[int, int]) -> str:
    """The clip at `fps`, fitted inside `size` without changing its shape.

    A dataset keeps one frame size, set by its first clip, and a clip of a
    different shape — a 16:9 take merged onto a 4:3 version — was stretched to
    it, squashing the hand it shows. It is scaled to fit and the rest padded
    black instead; a clip already the right shape is only scaled.
    """
    h, w = size
    return (
        f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease:flags=bicubic,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )


def _normalize(src: Path, fps: int, size: tuple[int, int], work: Path) -> Path:
    out = work / f"norm_{uuid.uuid4().hex}.mp4"
    cmd = [
        ffmpeg_exe(), "-y", "-i", str(src),
        "-vf", frame_filter(fps, size),
        "-an", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def iter_frames(src: Path, fps: int, size: tuple[int, int], work: Path) -> Iterator["object"]:
    import cv2

    norm = _normalize(src, fps, size, work)
    cap = cv2.VideoCapture(str(norm))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
        # The re-encoded copy has no use once its frames are read. It was left
        # behind, so a fifty-episode build kept fifty of them alongside fifty
        # originals for the whole run.
        with contextlib.suppress(OSError):
            norm.unlink()
