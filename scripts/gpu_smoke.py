"""Check, once, that the hand-tracking models run on this machine — before
the deploy that turns them on (docs/HAND2ROBOT_2.md, the DGX smoke test).

    docker compose run --rm --entrypoint python celery-compute scripts/gpu_smoke.py

Changes nothing: no queue, no storage, no file written. It loads each model
the settings name (RETARGET_DEPTH_MODEL, RETARGET_HAND_MODEL), runs it on one
made-up frame, and prints where it ran, how long it took and the GPU memory it
held — the numbers to put next to the AI model's before turning it on.
Exit status 0 when every named model ran.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def main() -> int:
    from dfpipeline import gpu
    from dfpipeline.config import settings

    print("PIPELINE_GPU:", settings.pipeline_gpu, "| GPU_MEMORY_FRACTION:", settings.gpu_memory_fraction)
    device = gpu.torch_device()
    print("device:", device)
    try:
        import torch

        print("torch", torch.__version__, "| CUDA", torch.version.cuda,
              "|", torch.cuda.get_device_name(0) if device == "cuda" else "no GPU")
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print("torch: not importable:", exc)
        return 1

    frame = (np.random.default_rng(0).uniform(0, 255, (720, 1280, 3))).astype(np.uint8)
    ok = True

    def held() -> str:
        if device != "cuda":
            return "-"
        return f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MB peak"

    depth_name = (settings.retarget_depth_model or "off").strip().lower()
    if depth_name not in ("", "off", "none"):
        from dfpipeline.retarget import scene

        started = time.perf_counter()
        model = scene.depth_model()
        if model is None:
            print(f"depth model {depth_name}: FAILED to load (see the log above)")
            ok = False
        else:
            loaded = time.perf_counter()
            depth, _ = model.infer(frame, 70.0)
            for _ in range(2):
                model.infer(frame, 70.0)
            done = time.perf_counter()
            print(f"depth model {depth_name}: OK — load {loaded - started:.1f}s, "
                  f"{(done - loaded) / 3:.2f}s a frame, depth {np.nanmin(depth):.2f}-{np.nanmax(depth):.2f} m, {held()}")

    if (settings.retarget_hand_model or "mediapipe").strip().lower() == "wilor":
        try:
            from dfpipeline.retarget import hand_wilor

            started = time.perf_counter()
            pipe = hand_wilor._pipeline()
            loaded = time.perf_counter()
            for _ in range(3):
                pipe.predict(frame)
            done = time.perf_counter()
            print(f"hand model wilor: OK — load {loaded - started:.1f}s, {(done - loaded) / 3:.2f}s a frame, {held()}")
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            print("hand model wilor: FAILED:", repr(exc)[:400])
            ok = False

    print("result:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
