# The DataForge pipeline, for the pipeline server: Celery workers for the heavy
# work — the AI check, hand tracking, retargeting, dataset builds — during the
# pipeline's hours. No client, no API, no database.
#
# A project of its own: build from this folder, nothing outside it is needed.
#   docker build -t dataforge-pipeline .
#
# The CPU's cores do most of the work; an NVIDIA GPU, where the server has one,
# takes hand tracking and the datasets' video encoding (dfpipeline/gpu.py,
# docker-compose.gpu.yaml — deploy.sh adds it on its own). The image is the
# same either way: the GPU's libraries come from the driver, mounted in by
# Docker's NVIDIA runtime. The image builds for the server's own architecture
# — x86_64 or arm64 (see README.md).

# -------------------------------------------------------------- the pipeline
FROM python:3.12-slim AS pipeline

# ffmpeg is not optional: every check reads frames with it and every export
# re-encodes through it.
#
# libglib2.0-0 is for opencv. libgl1 is deliberately absent: it was here for
# MediaPipe, which requires the desktop opencv build, and it drags in Mesa,
# LLVM and the whole X11 chain — most of a 135MB, 215-package apt step for a
# container that will never open a window. The pip layer below swaps that
# opencv for its headless twin instead, which needs none of it.
#
# libegl1 and libgles2 are not that chain. MediaPipe's native library links
# against libEGL.so.1 and libGLESv2.so.2 and will not load without them, even
# on the CPU delegate that never opens a GL context. `import mediapipe` does not
# load it — the first detector does — so without these every track.clip
# job fails with "libEGL.so.1: cannot open shared object file". This does bring
# Mesa and LLVM back: Debian's libegl1 hard-depends on libegl-mesa0, which no
# --no-install-recommends can drop. That is the price of object tracking running.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The hand-tracking models (docs/HAND2ROBOT_2.md): none by default, and the
# image is what it always was. For the DGX with its GPU:
#   docker build --build-arg PERCEPTION="depth" \
#                --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 .
# PERCEPTION is what scripts/install_perception.sh installs ("depth",
# "depth wilor"); TORCH_INDEX is where PyTorch comes from — its CUDA build,
# for the server's own architecture, when the models are to run on the GPU.
ARG PERCEPTION=""
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# torch first, and the CPU build of it (unless TORCH_INDEX says otherwise). `[export]` pulls lerobot, which
# requires torch, and the default Linux wheel on PyPI is the CUDA one: it
# carries the whole nvidia runtime, gigabytes of it, for work that never uses
# a GPU — even on a server that has one. Installing the CPU wheel from
# PyTorch's own index first means lerobot finds torch already satisfied and
# never asks PyPI for the CUDA one.
#
# It is also its own layer, and depends on nothing in the repository, so it is
# built once and then reused by every later deploy.
RUN pip install --no-cache-dir --index-url "$TORCH_INDEX" \
    "torch>=2.7,<2.12" "torchvision>=0.22,<0.27"

# Then the rest of the dependencies, from pyproject.toml alone, so this layer
# is rebuilt only when that file changes rather than on every edit to every
# source file. lerobot and its tree are not large next to torch, but they are
# not free either, and a one-line fix should not reinstall them.
#
# The stub is what makes it possible: setuptools finds packages by looking, so
# a wheel cannot be built from pyproject.toml on its own. An empty package is
# enough to resolve and install the dependencies, and is thrown away directly
# afterwards.
COPY pyproject.toml ./
RUN mkdir -p dfpipeline && touch dfpipeline/__init__.py \
    && pip install --no-cache-dir \
        --extra-index-url "$TORCH_INDEX" ".[ml]" \
    && rm -rf dfpipeline *.egg-info

# Three hundred megabytes of opencv arrive as two whole copies: lerobot asks
# for opencv-python-headless, MediaPipe asks for opencv-contrib-python, and
# both ship the same `cv2` package, so one silently overwrites the other on
# disk. The survivor is MediaPipe's desktop build — which is what dragged
# libgl1, Mesa, LLVM and X11 into the image.
#
# So both go, and the headless contrib build replaces them: same version, same
# contrib modules MediaPipe needs, no GUI and no OpenGL. pip will say
# mediapipe's pin is unsatisfied; it is satisfied in the only way that matters,
# by `import cv2` working. Nothing in this container opens a window.
#
# cmake goes with them. lerobot requires it to build, nothing calls it to run,
# and it is ninety megabytes in every container that ever starts.
#
# The import check is the point of doing this here: if the swap is ever wrong,
# the build fails, rather than the first export an operator is waiting on.
RUN pip uninstall -y opencv-contrib-python opencv-python-headless cmake \
    && pip install --no-cache-dir "opencv-contrib-python-headless==5.0.0.93" \
    && python -c "import cv2, mediapipe, scipy; print('cv2', cv2.__version__, 'headless — imports OK')"

# The two MediaPipe models the retargeting needs, fetched here rather than on
# the first job: a fresh container would otherwise pay for the download inside
# an export, and fail it outright on a box with no route to the internet. Its
# own layer, and the script it runs is stdlib only, so it is fetched once and
# reused by every later deploy.
#
# The SO-101 URDF is not here — it is committed
# under models/so101 and arrive with the source below.
COPY scripts/download_retarget_weights.py ./scripts/
RUN python scripts/download_retarget_weights.py --dir /app/models

# The hand-tracking models, when asked for: their packages (pinned, without
# the demo and training dependencies) and their weights, here at build time
# so that no job ever downloads one. Then the import, so a wrong package
# fails the build rather than the first clip of the night.
COPY scripts/install_perception.sh scripts/download_perception_weights.py ./scripts/
RUN if [ -n "$PERCEPTION" ]; then \
        sh scripts/install_perception.sh $PERCEPTION \
        && python scripts/download_perception_weights.py --dir /app/models $PERCEPTION \
        && for part in $PERCEPTION; do case "$part" in \
             depth) python -c "from moge.model.v2 import MoGeModel; print('MoGe imports OK')" ;; \
             wilor) python -c "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; print('WiLoR imports OK')" ;; \
           esac; done; \
    fi

# The import check above cannot catch a missing system library: MediaPipe only
# loads its native code when a task is created. Creating the object detector
# here is what does, so a missing libEGL fails the build instead of every
# tracking job in production.
RUN python -c "from mediapipe.tasks import python as mp; from mediapipe.tasks.python import vision; \
vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions( \
base_options=mp.BaseOptions(model_asset_path='/app/models/efficientdet_lite0.tflite'))).close(); \
print('ObjectDetector loads OK')"

# Then the real source, and the project itself without touching its
# dependencies — they are in the layers above.
COPY dfpipeline/ ./dfpipeline/
COPY scripts/gpu_smoke.py ./scripts/
COPY models/so101/ ./models/so101/
RUN pip install --no-cache-dir --no-deps . \
    && python -c "import dfpipeline.worker, dfpipeline.compute, dfpipeline.scheduler, dfpipeline.validation.judge; print('pipeline imports OK')"

# Staging and datasets live on volumes, not in the image. Without this the
# directories are created root-owned on first write and the unprivileged user
# below cannot use them.
RUN mkdir -p /app/_datasets /app/models \
    && useradd --system --uid 10001 --home /app robocollect \
    && chown -R robocollect:robocollect /app

# No database here, on purpose — the project has no database code at all.
ENV PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    EXPORT_DATA_DIR=/app/_datasets \
    RETARGET_MODELS_DIR=/app/models \
    ROLE=scheduled

# Media pulled from the Hub is cached, and the default puts that cache in
# the home directory - part of the container, not a volume. An export of a
# few hundred episodes fills it, nothing ever empties it, and it counts
# against the same disk the image layers and the build cache live on. On
# the volume it is visible, it survives a restart, and it can be cleared.
ENV HF_HOME=/app/_datasets/.hf-cache

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chown robocollect:robocollect /usr/local/bin/entrypoint.sh

USER robocollect

# No HTTP port and no healthcheck: these are queue workers. Whether they are
# alive is on the app server's Monitor page, which asks Celery which workers answer.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
