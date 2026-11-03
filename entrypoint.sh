#!/bin/bash
# Start one part of the pipeline, chosen by ROLE.
#
#   ROLE=scheduled  (the default) the workers, only inside PIPELINE_HOURS — one
#                   per kind of work, each with its own limit
#                   (dfpipeline/scheduler.py)
#   ROLE=celery     one worker for CELERY_QUEUES, always on — for testing
set -e

case "${ROLE:-scheduled}" in
    scheduled)
        echo "==> pipeline workers, during ${PIPELINE_HOURS:-22:00-09:00}"
        exec python -m dfpipeline.scheduler
        ;;
    celery)
        queues="${CELERY_QUEUES:-check,track,build}"
        echo "==> pipeline worker for: $queues"
        exec celery -A dfpipeline.worker worker -Q "$queues" \
            -c "${CELERY_CONCURRENCY:-8}" -n "${NODE_NAME:-pipeline}-${queues//,/-}@%h" \
            --loglevel=INFO
        ;;
    *)
        echo "!! unknown ROLE '$ROLE' (scheduled | celery)"
        exit 64
        ;;
esac
