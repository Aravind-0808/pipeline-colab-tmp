#!/bin/bash
# Build and start the pipeline on the pipeline server, from this folder.
#
#   ./deploy.sh
#
# Built here rather than pulled: the image is then always for this machine's
# own architecture (x86_64 or arm64), and no registry is involved.
#
# When the app server changes too, run this first: new pipeline code understands
# work queued by older app code, not the other way round (CONTRACT.md).
#
# Nothing here holds data that cannot be rebuilt except the staging folder,
# which is a volume and survives the restart; a task interrupted mid-run is
# handed to another worker by Celery, and its job's lease on the app server covers
# the gap.
set -euo pipefail

cd "$(dirname "$0")"
# The production settings: .env.production, or a plain .env on an older server.
ENV_FILE=.env.production
[ -f "$ENV_FILE" ] || ENV_FILE=.env
[ -f "$ENV_FILE" ] || { echo "!! no .env.production here — copy .env.example and fill it in"; exit 1; }
COMPOSE="docker compose -f docker-compose.yaml --env-file $ENV_FILE"

# The GPU, when this server has one Docker can hand to a container: hand
# tracking and dataset encoding then use it (dfpipeline/gpu.py). PIPELINE_GPU=off
# in the settings file keeps it on the CPU anyway.
if ! grep -qiE '^PIPELINE_GPU=(on|auto)' "$ENV_FILE" 2>/dev/null; then
    # The GPU belongs to the models and training here; PIPELINE_GPU=on to share it.
    echo "==> GPU: off (set PIPELINE_GPU=on to use it)"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 \
        && docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    COMPOSE="$COMPOSE -f docker-compose.gpu.yaml"
else
    echo "==> GPU: none Docker can use here — the CPU (install the NVIDIA Container Toolkit to use one)"
fi

echo "==> building for $(uname -m)"
$COMPOSE build

echo "==> starting"
$COMPOSE up -d --remove-orphans

echo "==> waiting for the workers to stay up"
sleep 20
running=$($COMPOSE ps --status running --services)
if grep -q celery-compute <<<"$running"; then
    $COMPOSE ps
    docker image prune -f >/dev/null 2>&1 || true
    echo "==> pipeline is up. The app server's Monitor page should now list this machine."
    exit 0
fi

echo "!! a service is not running. Recent logs:"
$COMPOSE logs --tail 60 || true
exit 1
