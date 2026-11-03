"""The pipeline's hours: when it works, and when it is asleep."""

import subprocess
from datetime import datetime, timezone

import pytest

from dfpipeline import hours, scheduler

IST = "Asia/Kolkata"


def at(hh: int, mm: int = 0) -> datetime:
    """A moment given in India time, as UTC (IST is UTC+5:30)."""
    minutes = (hh * 60 + mm - 330) % (24 * 60)
    return datetime(2026, 10, 12, minutes // 60, minutes % 60, tzinfo=timezone.utc)


@pytest.mark.parametrize("hh, open_", [(21, False), (22, True), (23, True), (0, True), (3, True),
                                       (7, True), (8, False), (12, False)])
def test_a_window_across_midnight(hh, open_):
    assert hours.is_open("22:00-08:00", IST, at(hh)) is open_


def test_no_window_means_always():
    assert hours.is_open("", IST, at(12)) is True
    assert hours.is_open("nonsense", IST, at(12)) is True
    assert hours.seconds_until_close("", IST, at(12)) is None


def test_how_long_until_it_opens_and_closes():
    assert hours.seconds_until_open("22:00-08:00", IST, at(21, 30)) == 30 * 60
    assert hours.seconds_until_open("22:00-08:00", IST, at(23)) == 0
    assert hours.seconds_until_close("22:00-08:00", IST, at(7)) == 60 * 60
    assert hours.seconds_until_close("22:00-08:00", IST, at(12)) == 0


class FakeProc:
    def __init__(self, cmd):
        self.cmd, self.signals, self.code = cmd, [], None

    @property
    def returncode(self):
        return self.code

    def poll(self):
        return self.code

    def send_signal(self, sig):
        self.signals.append(sig)
        self.code = 0  # finishes when asked

    def wait(self, _t=None):
        return self.code

    def kill(self):
        self.code = -9


@pytest.fixture
def sched(monkeypatch):
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **_: started.append(FakeProc(cmd)) or started[-1])
    monkeypatch.setattr(scheduler.settings, "stop_grace_sec", 0)
    s = scheduler.Scheduler()
    s.started = started
    return s


def test_inside_the_hours_one_limited_worker_per_kind(sched, monkeypatch):
    monkeypatch.setattr(sched, "open_now", lambda: True)
    sched.tick()
    queues = {" ".join(p.cmd).split(" -Q ")[1].split()[0]: p.cmd for p in sched.started}
    assert set(queues) == {"check", "track", "build"}
    assert queues["build"][queues["build"].index("-c") + 1] == "1"
    sched.tick()  # already running: nothing more started
    assert len(sched.started) == 3


def test_outside_the_hours_they_are_asked_to_finish(sched, monkeypatch):
    monkeypatch.setattr(sched, "open_now", lambda: True)
    sched.tick()
    monkeypatch.setattr(sched, "open_now", lambda: False)
    sched.tick()
    import signal

    assert all(p.signals == [signal.SIGTERM] for p in sched.started)
    assert sched.workers == {}


def test_a_worker_that_dies_inside_the_hours_is_started_again(sched, monkeypatch):
    monkeypatch.setattr(sched, "open_now", lambda: True)
    sched.tick()
    sched.started[0].code = 1
    sched.tick()
    assert len(sched.started) == 4


@pytest.mark.parametrize("hh, mm, open_", [(21, 59, False), (22, 0, True), (2, 0, True),
                                           (8, 30, True), (8, 59, True), (9, 0, False), (14, 0, False)])
def test_the_default_hours_are_ten_at_night_to_nine_in_the_morning_india_time(hh, mm, open_):
    from dfpipeline.config import Settings

    defaults = Settings(_env_file=None, pipeline_hours=Settings.model_fields["pipeline_hours"].default)
    assert (defaults.pipeline_hours, defaults.pipeline_tz) == ("22:00-09:00", IST)
    assert hours.is_open(defaults.pipeline_hours, defaults.pipeline_tz, at(hh, mm)) is open_


def test_ist_is_india_time():
    """'IST' is not a zoneinfo key, and a setting of PIPELINE_TZ=IST crashed the scheduler."""
    for tz in ("IST", "ist", "Asia/Kolkata"):
        assert hours.is_open("22:00-09:00", tz, at(8, 30))
        assert not hours.is_open("22:00-09:00", tz, at(9, 0))


def test_the_workers_keep_to_a_share_of_the_machine(monkeypatch):
    """The DGX also serves the models and runs training: every math library is
    held to a few cores per process, and worker processes are replaced before
    a leak can add up."""
    from dfpipeline import scheduler, worker

    monkeypatch.setattr(scheduler.settings, "cpu_threads", 2)
    env = scheduler.worker_env()
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "2"
    assert worker.celery_app.conf.worker_max_tasks_per_child == 50
    assert worker.celery_app.conf.worker_max_memory_per_child == 3000 * 1024
    assert scheduler.settings.pipeline_gpu == "off"  # the GPU is the models', unless asked


def test_the_container_has_a_memory_and_cpu_cap():
    from pathlib import Path

    import yaml

    compose = yaml.safe_load((Path(__file__).parents[1] / "docker-compose.yaml").read_text())
    service = compose["services"]["celery-compute"]
    assert service["mem_limit"] == service["memswap_limit"] == "${PIPELINE_MEM_LIMIT:-16g}"
    assert service["cpus"] == "${PIPELINE_CPUS:-8}"


def test_the_share_has_defaults_as_config_keys():
    from dfpipeline.config import Settings

    s = Settings(_env_file=None)
    assert (s.pipeline_mem_limit, s.pipeline_cpus, s.cpu_threads) == ("16g", 8.0, 2)
    assert (s.worker_max_tasks, s.worker_max_memory_mb, s.check_concurrency) == (50, 3000, 2)
