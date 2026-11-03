"""The GPU: where hand tracking runs — there is no CPU path for it.

Two things here use it:

- the PyTorch models of hand tracking (the depth model, scene.py; WiLoR,
  hand_wilor.py), within a fixed share of the GPU's memory (`torch_device`).
  They need it: without a CUDA device `torch_device` raises and the clip
  fails, rather than running for hours on the CPU;
- the dataset's video encoding (NVENC, H.264, the codec two-arm SO-101
  datasets such as Qiu-Xinchuan/so101_towel are in), where it works.

The arm is solved by small least-squares problems, one a frame, that a GPU
would only slow down; the object detector is MediaPipe's, on the CPU.

NVENC is not taken on trust: FFmpeg lists it whether or not a GPU is there
(LeRobot's own `vcodec="auto"` picks it on a laptop with none, and the build
fails), so one small frame is really encoded first.

`PIPELINE_GPU` (video encoding only): `off` (the default: never), `auto`,
`on` (try even without an NVIDIA driver, and warn when falling back).
"""

from __future__ import annotations

import functools
import io
import logging
import shutil
from pathlib import Path

from dfpipeline.config import settings

log = logging.getLogger("pipeline.gpu")


def _mode() -> str:
    return (settings.pipeline_gpu or "auto").strip().lower()


@functools.cache
def nvidia_present() -> bool:
    """An NVIDIA driver on this machine (or in this container)."""
    return Path("/proc/driver/nvidia/version").exists() or shutil.which("nvidia-smi") is not None


def wanted() -> bool:
    mode = _mode()
    if mode == "off":
        return False
    return mode == "on" or nvidia_present()


def _fell_back(what: str, why: str) -> None:
    level = logging.WARNING if _mode() == "on" else logging.INFO
    log.log(level, "%s on the CPU: the GPU did not work (%s)", what, why)


# ------------------------------------------------------------------ PyTorch

@functools.cache
def torch_device() -> str:
    """"cuda" for the PyTorch models; raises where there is no working GPU.

    This process may hold only GPU_MEMORY_FRACTION of the GPU's memory: the
    DGX's GPU serves the AI model and training first, and a model that grew
    into the rest would take them down. Set once, before the first
    allocation, which is why it is here and cached.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("hand tracking needs a CUDA GPU, and PyTorch sees none "
                           f"(torch {torch.__version__}, CUDA build {torch.version.cuda})")
    torch.cuda.set_per_process_memory_fraction(settings.gpu_memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = True
    log.info("PyTorch models on the GPU, within %.0f%% of its memory", 100 * settings.gpu_memory_fraction)
    return "cuda"


# ------------------------------------------------------------- video encoding

#: The NVENC encoder, and the software one, for each codec.
_NVENC = {"h264": "h264_nvenc", "hevc": "hevc_nvenc"}
_SOFTWARE = {"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}


@functools.cache
def nvenc_works(encoder: str = "h264_nvenc") -> bool:
    """Whether this NVENC encoder really encodes here: one small frame, for real."""
    if not wanted():
        return False
    try:
        import av
        import numpy as np

        with av.open(io.BytesIO(), "w", format="mp4") as out:
            stream = out.add_stream(encoder, rate=30)
            stream.width, stream.height, stream.pix_fmt = 256, 144, "yuv420p"
            frame = av.VideoFrame.from_ndarray(np.zeros((144, 256, 3), np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
    except Exception as exc:  # noqa: BLE001 - no GPU, no driver, no NVENC: the CPU
        _fell_back(f"video encoding ({encoder})", str(exc)[:300])
        return False
    log.info("video encoding on the GPU (%s)", encoder)
    return True


def video_codec(base_codec: str | None = None) -> str:
    """The encoder for a dataset's videos.

    A new dataset: H.264 on the GPU's NVENC where it works, else LeRobot's
    default, AV1 on the CPU. A new version merged onto an older one keeps
    that version's codec — LeRobot joins the two by concatenating the video
    files, which needs one codec — on the GPU if it has an encoder for it.
    """
    if base_codec:
        codec = {"libsvtav1": "av1", "libx264": "h264", "libx265": "hevc"}.get(base_codec, base_codec)
        hardware = _NVENC.get(codec)
        if hardware and nvenc_works(hardware):
            return hardware
        return _SOFTWARE.get(codec, "libsvtav1")
    return "h264_nvenc" if nvenc_works("h264_nvenc") else "libsvtav1"
