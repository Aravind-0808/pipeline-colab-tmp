"""Run the pipeline's workers only inside its hours (PIPELINE_HOURS).

    python -m dfpipeline.scheduler

The DGX is the pipeline's at night only. This process starts one Celery worker
per kind of work when the window opens and stops them when it closes:

    check   the AI check       CHECK_CONCURRENCY tasks at once
    track   hand tracking      TRACK_CONCURRENCY
    build   datasets, publish  BUILD_CONCURRENCY

Never everything at once: each kind has its own worker and its own limit, so
a long dataset build cannot take the slots the night's checks need, and a
burst of checks cannot starve tracking.

At the end of the window each worker is asked to finish what it is doing
(warm shutdown, SIGTERM) and given STOP_GRACE_SEC. Whatever is still running
then is stopped (SIGQUIT). Nothing is lost: tasks are acknowledged only when
they finish (task_acks_late), so an unfinished one stays in Redis and is handed
out again — the next night, when a worker starts. A worker that dies inside the
window is started again.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time

from dfpipeline import heartbeat, hours
from dfpipeline.config import settings

log = logging.getLogger("pipeline.scheduler")

TICK_SEC = 30


def perception_models() -> bool:
    """Hand tracking uses PyTorch models (docs/HAND2ROBOT_2.md)."""
    return (settings.retarget_hand_model or "mediapipe").strip().lower() != "mediapipe" or \
        (settings.retarget_depth_model or "off").strip().lower() not in ("", "off", "none")


def groups() -> list[tuple[str, int]]:
    """(queue, tasks at once) for each worker this machine runs.

    With hand-tracking models, `track` runs one clip at a time: each worker
    process loads its own copy of every model, so two at once is twice the
    GPU memory, and the GPU is the AI model's first.
    """
    track = 1 if perception_models() else max(1, settings.track_concurrency)
    return [
        ("check", max(1, settings.check_concurrency)),
        ("track", track),
        ("build", max(1, settings.build_concurrency)),
    ]


def command(queue: str, concurrency: int) -> list[str]:
    node = settings.node_name or "pipeline"
    cmd = [
        sys.executable, "-m", "celery", "-A", "dfpipeline.worker", "worker",
        "-Q", queue, "-c", str(concurrency), "-n", f"{node}-{queue}@%h",
        "--loglevel=INFO",
    ]
    if queue == "track" and perception_models():
        # A process holding the models is past the usual memory bar from its
        # first clip, and would be replaced — and reload them — after every
        # one. Its own, higher bar; the models are what it is for.
        cmd.append(f"--max-memory-per-child={max(1, settings.perception_worker_memory_mb) * 1024}")
    return cmd


def worker_env() -> dict[str, str]:
    """The workers' environment, with every math library held to its share of
    cores: by default each one takes them all, in every process at once."""
    threads = str(max(1, settings.cpu_threads))
    caps = {k: threads for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    return {**os.environ, **caps}


class Scheduler:
    def __init__(self) -> None:
        self.workers: dict[str, subprocess.Popen] = {}
        self.stopping = False
        self.window, self.tz = settings.pipeline_hours or "", settings.pipeline_tz

    def open_now(self) -> bool:
        # As set on the app's Settings page (heartbeat.working_hours).
        self.window, self.tz = heartbeat.working_hours()
        return hours.is_open(self.window, self.tz)

    def start(self) -> None:
        for queue, concurrency in groups():
            proc = self.workers.get(queue)
            if proc is not None and proc.poll() is None:
                continue
            if proc is not None:
                log.warning("the %s worker exited (%s); starting it again", queue, proc.returncode)
            self.workers[queue] = subprocess.Popen(command(queue, concurrency), env=worker_env())
            log.info("started the %s worker, %d at once (container: %s memory, %s cores; %d threads each)",
                     queue, concurrency, settings.pipeline_mem_limit, settings.pipeline_cpus,
                     settings.cpu_threads)

    def stop(self) -> None:
        """Ask every worker to finish, then stop what is left after the grace."""
        running = {q: p for q, p in self.workers.items() if p.poll() is None}
        if not running:
            self.workers.clear()
            return
        log.info("the pipeline's hours are over: finishing %s", ", ".join(running))
        for proc in running.values():
            proc.send_signal(signal.SIGTERM)  # warm: finish the task in hand
        deadline = time.monotonic() + settings.stop_grace_sec
        while time.monotonic() < deadline and any(p.poll() is None for p in running.values()):
            time.sleep(2)
        for queue, proc in running.items():
            if proc.poll() is None:
                log.warning("the %s worker did not finish in %ss; stopping it — its task "
                            "goes back to the queue for the next night", queue, settings.stop_grace_sec)
                proc.send_signal(signal.SIGQUIT)  # cold: the task is not acknowledged
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.workers.clear()

    def tick(self) -> None:
        if self.open_now():
            self.start()
        elif self.workers:
            self.stop()

    def run(self) -> None:
        def _exit(*_):
            self.stopping = True

        signal.signal(signal.SIGTERM, _exit)
        signal.signal(signal.SIGINT, _exit)
        log.info("pipeline hours %s (%s), %s", settings.pipeline_hours or "always",
                 settings.pipeline_tz, ", ".join(f"{q}×{c}" for q, c in groups()))
        was_asleep = None
        seen_hours = None
        while not self.stopping:
            self.tick()
            if (self.window, self.tz) != seen_hours:
                log.info("pipeline hours now %s (%s)", self.window or "always", self.tz)
                seen_hours = (self.window, self.tz)
            asleep = not self.workers
            # "Here", every tick, awake or asleep: the app's Monitor reads it.
            heartbeat.beat(asleep, self.window, self.tz)
            if asleep and was_asleep is not True:
                wait = hours.seconds_until_open(self.window, self.tz)
                log.info("asleep; the pipeline's hours start in %dh%02dm", wait // 3600, wait % 3600 // 60)
            was_asleep = asleep
            for _ in range(TICK_SEC):
                if self.stopping:
                    break
                time.sleep(1)
        self.stop()


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    Scheduler().run()


if __name__ == "__main__":
    main()
