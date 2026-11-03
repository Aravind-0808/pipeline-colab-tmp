"""The GPU where an NVIDIA one works, the CPU everywhere else (dfpipeline/gpu.py)."""

import json
import subprocess

import pytest

from dfpipeline import gpu
from dfpipeline.config import settings


@pytest.fixture(autouse=True)
def fresh():
    """Each test decides afresh; the answers are cached per process."""
    def clear():
        for cached in (gpu.nvidia_present, gpu.hand_landmarker_on_gpu, gpu.nvenc_works):
            getattr(cached, "cache_clear", lambda: None)()

    clear()
    yield
    clear()


def _machine(monkeypatch, mode: str, nvidia: bool, nvenc: bool = True):
    monkeypatch.setattr(settings, "pipeline_gpu", mode)
    monkeypatch.setattr(gpu, "nvidia_present", lambda: nvidia)
    monkeypatch.setattr(gpu, "nvenc_works", lambda encoder="h264_nvenc": gpu.wanted() and nvenc)


def test_auto_uses_the_gpu_only_where_an_nvidia_driver_is(monkeypatch):
    _machine(monkeypatch, "auto", nvidia=False)
    assert not gpu.wanted()  # Mesa's software OpenGL "works", slower than the CPU
    _machine(monkeypatch, "auto", nvidia=True)
    assert gpu.wanted()


def test_off_is_never_and_on_is_always_tried(monkeypatch):
    _machine(monkeypatch, "off", nvidia=True)
    assert not gpu.wanted()
    assert gpu.video_codec() == "libsvtav1"
    _machine(monkeypatch, "on", nvidia=False)
    assert gpu.wanted()


def test_a_new_dataset_is_h264_on_the_gpu_and_av1_without(monkeypatch):
    _machine(monkeypatch, "auto", nvidia=True)
    assert gpu.video_codec() == "h264_nvenc"
    _machine(monkeypatch, "auto", nvidia=True, nvenc=False)  # a driver without NVENC
    assert gpu.video_codec() == "libsvtav1"
    _machine(monkeypatch, "auto", nvidia=False)
    assert gpu.video_codec() == "libsvtav1"


def test_a_new_version_keeps_the_codec_of_the_one_it_merges_onto(monkeypatch):
    _machine(monkeypatch, "auto", nvidia=True)
    assert gpu.video_codec("av1") == "libsvtav1"  # no AV1 encoder on the GPU: the CPU's
    assert gpu.video_codec("h264") == "h264_nvenc"
    _machine(monkeypatch, "auto", nvidia=False)
    assert gpu.video_codec("h264") == "libx264"  # an H.264 dataset grows in H.264 on the CPU too


def test_hand_tracking_tries_the_gpu_in_a_child_process(monkeypatch):
    """A native abort on the GPU kills the process it happens in: the try is
    a child's, and the worker carries on, on the CPU."""
    _machine(monkeypatch, "auto", nvidia=True)
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], -6, "", "F0000 Check failed: something native"))
    assert gpu.hand_landmarker_on_gpu("model.task") is False

    gpu.hand_landmarker_on_gpu.cache_clear()
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "ok\n", ""))
    assert gpu.hand_landmarker_on_gpu("model.task") is True


def test_without_a_gpu_nothing_is_even_tried(monkeypatch):
    _machine(monkeypatch, "auto", nvidia=False)

    def never(*a, **k):
        raise AssertionError("no child process without a GPU")

    monkeypatch.setattr(gpu.subprocess, "run", never)
    assert gpu.hand_landmarker_on_gpu("model.task") is False


def test_the_codec_of_a_built_version_is_read_from_its_info(tmp_path):
    from dfpipeline.lerobot_builder import _video_codec

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"features": {
        "observation.images.top": {"dtype": "video", "info": {"video.codec": "av1"}}}}))
    assert _video_codec(tmp_path, "observation.images.top") == "av1"
    assert _video_codec(tmp_path / "missing", "observation.images.top") is None
