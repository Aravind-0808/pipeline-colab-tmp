"""The Celery worker: takes `pipeline.compute` from the app server's Redis.

    celery -A dfpipeline.worker worker -Q check,track,build -c 4

In production these are started and stopped by dfpipeline/scheduler.py, one
per queue, only during the pipeline's hours.

Each task arrives as the first link of a chain the app server built; Celery passes
its result on to the second link, `pipeline.finish`, which runs on the app server
(queue "db") and writes it. That task is not defined here — Celery sends it by
name — and neither is anything that touches a database.
"""

from __future__ import annotations

import logging
import os

from celery import Celery, signals

from dfpipeline.config import settings

log = logging.getLogger("pipeline")

DB_QUEUE = "db"

celery_app = Celery("dataforge")
celery_app.conf.update(
    # The same settings as the app server's side of the queue (CONTRACT.md).
    broker_url=settings.redis_url or "memory://",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    broker_transport_options={"visibility_timeout": 9 * 3600},
    task_default_queue="default",
    task_serializer="json",
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    worker_hijack_root_logger=False,
    # A worker process is replaced after so many tasks, or once it holds this
    # much memory: OpenCV and MediaPipe leak a little per clip, and a night of
    # clips would otherwise add it all up on a machine shared with the models.
    worker_max_tasks_per_child=max(1, settings.worker_max_tasks),
    worker_max_memory_per_child=max(1, settings.worker_max_memory_mb) * 1024,  # in KB
)


@signals.worker_process_init.connect
def _hold_to_a_share_of_the_cores(**_kwargs) -> None:
    """OpenCV starts a thread per core in every process unless told otherwise."""
    try:
        import cv2

        cv2.setNumThreads(max(1, settings.cpu_threads))
    except Exception:  # noqa: BLE001, S110 - no OpenCV, nothing to hold
        pass


@celery_app.task(name="pipeline.compute")
def compute_task(name: str, spec: dict, protocol: int | None = None) -> dict:
    """One job's heavy half. Its result goes on along the chain to the app server."""
    from dfpipeline import compute

    return compute.run(name, spec, protocol)


def send_progress(export_id: str, fraction: float, step: str) -> None:
    """An export's progress, to the app server's "db" queue. Never retried."""
    celery_app.send_task(
        "pipeline.export_progress", args=(export_id, fraction, step), queue=DB_QUEUE, retry=False
    )


@signals.worker_ready.connect
def _sweep_unfinished_builds(**_kwargs) -> None:
    """Clear export workspaces a restart left half-built. A worker starting is
    the one moment nothing on this machine is building."""
    try:
        from dfpipeline.compute import sweep_stale_builds

        swept = sweep_stale_builds()
        if swept["removed"]:
            log.info(
                "startup: removed %d unfinished export build(s), %.1f MB",
                swept["removed"], swept["freed_bytes"] / 1e6,
            )
    except Exception:
        log.exception("startup sweep failed; carrying on")


@signals.worker_ready.connect
def _hand_back_abandoned(**_kwargs) -> None:
    """Checks and tracking a stopped worker left unfinished go back to their
    queue now, not after the nine-hour visibility timeout. See recovery.py."""
    try:
        from dfpipeline import recovery

        recovery.start(celery_app)
    except Exception:
        log.exception("could not start the hand-back sweep; carrying on")


@signals.worker_ready.connect
def _say_where(**_kwargs) -> None:
    if not settings.redis_url:
        log.warning("REDIS_URL is not set: this worker has no app server to take work from")
    log.info("pipeline worker on %s, protocol v%s",
             settings.node_name or os.environ.get("HOSTNAME", "?"),
             __import__("dfpipeline").PROTOCOL_VERSION)
