"""The GPU: hand tracking only on it, video encoding where it works (dfpipeline/gpu.py)."""

import json

import pytest

from dfpipeline import gpu
from dfpipeline.config import settings


@pytest.fixture(autouse=True)
def fresh():
    """Each test decides afresh; the answers are cached per process."""
    def clear():
        for cached in (gpu.nvidia_present, gpu.nvenc_works):
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
    assert not gpu.wanted()
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


def test_hand_tracking_has_no_cpu_path(monkeypatch):
    """Without a CUDA device the models refuse, rather than run for hours on the CPU."""
    torch = pytest.importorskip("torch")
    gpu.torch_device.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs a CUDA GPU"):
        gpu.torch_device()
    gpu.torch_device.cache_clear()


def test_the_codec_of_a_built_version_is_read_from_its_info(tmp_path):
    from dfpipeline.lerobot_builder import _video_codec

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"features": {
        "observation.images.top": {"dtype": "video", "info": {"video.codec": "av1"}}}}))
    assert _video_codec(tmp_path, "observation.images.top") == "av1"
    assert _video_codec(tmp_path / "missing", "observation.images.top") is None
