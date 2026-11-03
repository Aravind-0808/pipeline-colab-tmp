#!/bin/sh
# The hand-tracking models' packages (docs/HAND2ROBOT_2.md), pinned to exact
# commits, for the image (Dockerfile, PERCEPTION=...) and for Colab — the same
# commands in both, so what was measured on Colab is what runs on the DGX.
#
#   sh scripts/install_perception.sh depth          MoGe-2 (the depth model)
#   sh scripts/install_perception.sh depth wilor    and WiLoR (the hand model)
#
# PyTorch itself is not installed here: the image installs it first, from the
# index for its machine (TORCH_INDEX), and Colab has its own.
#
# --no-deps, on purpose. MoGe's own list asks for gradio, matplotlib, trimesh
# and the desktop opencv (which would undo the image's headless swap) for its
# demo and training, none of which inference imports; WiLoR-mini pins
# torch<=2.5, which has no build for the DGX Spark's GPU. What inference
# imports is listed here instead.
set -eu

MOGE="git+https://github.com/microsoft/MoGe.git@b942f00bdc2a2a23ebb474fbe034d487e6dcceec"         # MoGe 2.0.0
UTILS3D="git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"  # MoGe 2.0.0's pin
WILOR="git+https://github.com/warmshao/WiLoR-mini.git@ebec42f94c389070cdd7dda6fd1bf0b4a659c960"

for part in "$@"; do
    case "$part" in
        depth)
            pip install --no-cache-dir --no-deps "moge @ $MOGE" "utils3d @ $UTILS3D"
            pip install --no-cache-dir "huggingface_hub>=0.23"
            ;;
        wilor)
            # Experimental: verify on Colab first (docs/HAND2ROBOT_2.md, phase 1).
            pip install --no-cache-dir --no-deps "wilor-mini @ $WILOR" "smplx==0.1.28" \
                "ultralytics==8.1.34" "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
            pip install --no-cache-dir timm einops roma scikit-image "huggingface_hub>=0.23"
            ;;
        *)
            echo "unknown part: $part (depth | wilor)" >&2
            exit 2
            ;;
    esac
done
