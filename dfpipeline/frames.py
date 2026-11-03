"""Cheap frame sampling for clip inspection.

The expensive way to look at a clip is to download it and decode every frame.
We never do that: the header is read on its own to get the duration, a handful
of frames are pulled with input seeks (HTTP range requests when the source is
remote), and they are composed into a single timestamped contact sheet so one
image — not a video — reaches the model.
"""

from __future__ import annotations

import contextlib
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from dfpipeline.config import settings

_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")
_RESOLUTION = re.compile(r"Video:.*?(\d{2,5})x(\d{2,5})")


def ffmpeg_exe() -> str:
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class SourceUnavailable(OSError):
    """The clip could not be opened at all — a missing file, an HTTP error, a
    connection that failed. That says nothing about the recording: it is
    retried and left undecided, never judged "undecodable"."""


# What ffmpeg prints when it cannot open the input, as opposed to opening it
# and finding nothing it can decode.
_OPEN_FAILED = re.compile(
    r"No such file or directory|Server returned \d|HTTP error|Connection refused|Connection timed out|"
    r"Connection reset|Failed to resolve hostname|Network is unreachable|Input/output error|Operation timed out"
)


@dataclass(frozen=True)
class ClipProbe:
    duration: float | None
    width: int | None
    height: int | None

    @property
    def known(self) -> bool:
        return self.duration is not None and self.duration > 0


def probe(source: str) -> ClipProbe:
    """Read only the container header — ffmpeg exits non-zero after printing it."""
    out = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", source],
        capture_output=True,
        text=True,
        errors="replace",
    ).stderr
    duration = None
    if m := _DURATION.search(out):
        h, mm, ss = m.groups()
        duration = int(h) * 3600 + int(mm) * 60 + float(ss)
    width = height = None
    if m := _RESOLUTION.search(out):
        width, height = int(m.group(1)), int(m.group(2))
    if duration is None and width is None and (m := _OPEN_FAILED.search(out)):
        raise SourceUnavailable(f"ffmpeg could not open the clip: {m.group(0)}")
    return ClipProbe(duration=duration, width=width, height=height)


_TIME = re.compile(r"time=(\d+):(\d+):(\d+\.?\d*)")

# Used when the duration is unknowable: probe these offsets and keep what exists.
FALLBACK_OFFSETS = [0.5, 1.5, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]


def duration_by_decode(source: str, headers: dict | None = None) -> float | None:
    """Last resort: decode with no output and read the final timestamp.

    Clips recorded in the browser frequently carry no duration in the container
    header. Treating that as "corrupt" would auto-reject good work, so we spend
    one cheap decode rather than guess.
    """
    head_args = []
    if headers and source.startswith("http"):
        crlf = chr(13) + chr(10)
        head_args = ["-headers", "".join(f"{k}: {v}{crlf}" for k, v in headers.items())]
    out = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", *head_args, "-i", source, "-f", "null", "-"],
        capture_output=True, text=True, errors="replace",
    ).stderr
    stamps = _TIME.findall(out)
    if not stamps:
        return None
    h, m, sec = stamps[-1]
    total = int(h) * 3600 + int(m) * 60 + float(sec)
    return total or None


def sample_timestamps(duration: float, count: int) -> list[float]:
    """Evenly spaced instants, skipping the very first/last frames (often blank)."""
    if duration <= 0 or count <= 0:
        return []
    if count == 1:
        return [duration / 2]
    start, end = duration * 0.05, duration * 0.95
    step = (end - start) / (count - 1)
    return [round(start + i * step, 3) for i in range(count)]


@contextlib.contextmanager
def readable_source(url: str, size_bytes: int | None = None, headers: dict | None = None):
    """Yield something ffmpeg can seek.

    Small clips are fetched once (one round trip beats N ranged seeks); large
    ones are handed over as a URL so ffmpeg range-seeks and we never pull the
    whole file. A file already on this disk — a clip in local storage — is
    read where it is: there is nothing to fetch, and it is not a URL.
    """
    if "://" not in url:
        if not Path(url).is_file():
            raise SourceUnavailable(f"no file at {url}")
        yield url
        return
    if size_bytes is not None and size_bytes > settings.frame_download_max_bytes:
        yield url
        return
    tmp = Path(tempfile.mkdtemp()) / "clip"
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60, headers=headers or {}) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        yield str(tmp)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
            tmp.parent.rmdir()


def extract_frame(source: str, at: float, width: int, headers: dict | None = None) -> Image.Image | None:
    """One frame via input seek (`-ss` before `-i`) — fast and byte-thrifty."""
    head_args = []
    if headers and source.startswith("http"):
        crlf = chr(13) + chr(10)
        head_args = ["-headers", "".join(f"{k}: {v}{crlf}" for k, v in headers.items())]
    proc = subprocess.run(
        [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            *head_args,
            "-ss", f"{at:.3f}", "-i", source,
            "-frames:v", "1", "-vf", f"scale={width}:-2",
            "-f", "image2", "-c:v", "mjpeg", "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        img = Image.open(BytesIO(proc.stdout))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def extract_frames(source: str, timestamps: list[float], width: int,
                   headers: dict | None = None) -> list[tuple[float, Image.Image]]:
    frames = []
    for ts in timestamps:
        if (img := extract_frame(source, ts, width, headers)) is not None:
            frames.append((ts, img))
    return frames


def contact_sheet(frames: list[tuple[float, Image.Image]], columns: int = 3, label: str | None = None) -> Image.Image:
    """Compose frames into one labelled grid.

    A grid is dramatically cheaper in vision tokens than sending each frame as
    its own image, and the burnt-in timestamps let the model talk about *when*
    something happened.
    """
    if not frames:
        raise ValueError("no frames to compose")
    cell_w, cell_h = frames[0][1].size
    cols = min(columns, len(frames))
    rows = math.ceil(len(frames) / cols)
    header = 18 if label else 0
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + header), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    if label:
        draw.text((6, 4), label[:120], fill=(235, 235, 235))
    for i, (ts, img) in enumerate(frames):
        if img.size != (cell_w, cell_h):
            img = img.resize((cell_w, cell_h))
        x, y = (i % cols) * cell_w, header + (i // cols) * cell_h
        sheet.paste(img, (x, y))
        tag = f"{ts:.1f}s"
        draw.rectangle([x + 2, y + 2, x + 8 + 7 * len(tag), y + 16], fill=(0, 0, 0))
        draw.text((x + 5, y + 4), tag, fill=(255, 255, 255))
    return sheet


def to_data_url(image: Image.Image, quality: int = 70) -> str:
    import base64

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def inspect_clip(url: str, size_bytes: int | None, frame_count: int, width: int,
                 headers: dict | None = None, label: str | None = None):
    """probe -> sample -> extract -> compose.

    Returns (probe, frames, sheet). The frames come back too so the free checks
    can measure brightness/motion without decoding anything a second time.
    """
    with readable_source(url, size_bytes, headers) as source:
        return read_source(source, frame_count, width, headers=headers, label=label)


def read_source(source: str, frame_count: int, width: int,
                headers: dict | None = None, label: str | None = None):
    """The looking part, on something ffmpeg can already open.

    Split out so a clip that is still on this machine — one being uploaded
    right now — can be measured without being fetched back from storage first.
    """
    info = probe(source)
    duration = info.duration if info.known else duration_by_decode(source, headers)
    info = ClipProbe(duration=duration, width=info.width, height=info.height)

    # Unknown duration is not corruption: sample a fixed ladder instead, so the
    # clip still gets judged (the duration rules simply skip themselves).
    stamps = (
        sample_timestamps(duration, frame_count)
        if duration and duration > 0
        else FALLBACK_OFFSETS[:frame_count]
    )
    frames = extract_frames(source, stamps, width, headers)
    if not frames:
        return info, [], None
    return info, frames, contact_sheet(frames, label=label)
