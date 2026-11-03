"""Fetch the hand-tracking models' weights into the models folder, at build
time — never inside a job (docs/HAND2ROBOT_2.md).

    python scripts/download_perception_weights.py --dir /app/models depth wilor

The models then load from that folder (scene.MoGe, hand_wilor): a worker
never downloads, and a server with no route to Hugging Face still works.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# RETARGET_DEPTH_MODEL -> checkpoint (as dfpipeline/retarget/scene.py MoGe.CHECKPOINTS).
MOGE = {
    "moge-2-vits": "Ruicheng/moge-2-vits-normal",
    "moge-2-vitb": "Ruicheng/moge-2-vitb-normal",
    "moge-2-vitl": "Ruicheng/moge-2-vitl-normal",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./models")
    parser.add_argument("--depth-model", default="moge-2-vitb", choices=sorted(MOGE))
    parser.add_argument("parts", nargs="+", choices=["depth", "wilor"])
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    root = Path(args.dir)
    if "depth" in args.parts:
        target = root / "moge"
        target.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(MOGE[args.depth_model], "model.pt", local_dir=target / args.depth_model)
        print("depth model:", path)
    if "wilor" in args.parts:
        # WiLoR-mini reads <dir>/pretrained_models/{wilor_final.ckpt, detector.pt, MANO_RIGHT.pkl, ...}.
        path = snapshot_download("warmshao/WiLoR-mini", allow_patterns=["pretrained_models/*"],
                                 local_dir=root / "wilor")
        print("hand model:", path)


if __name__ == "__main__":
    main()
