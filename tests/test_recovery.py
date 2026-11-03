"""Short jobs a stopped worker left unfinished go back to their queue."""

import json

import fakeredis
import pytest

from dfpipeline import recovery

NOW = 1_000_000.0


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


def _taken(r, tag: str, task_id: str, queue: str, ago: float) -> None:
    """A message a worker took `ago` seconds ago and never acknowledged — as kombu stores it."""
    payload = {"body": "e30=", "headers": {"id": task_id, "task": "pipeline.compute"}, "properties": {}}
    r.hset(recovery.UNACKED, tag, json.dumps([payload, queue, queue]))
    r.zadd(recovery.UNACKED_INDEX, {tag: NOW - ago})


def test_a_check_no_worker_is_running_goes_back_to_its_queue(r):
    _taken(r, "t1", "check-1", "check", ago=600)
    assert recovery.hand_back(r, running=set(), now=NOW) == ["check-1"]
    assert r.llen("check") == 1
    assert json.loads(r.lindex("check", 0))["headers"]["id"] == "check-1"
    assert not r.hexists(recovery.UNACKED, "t1") and r.zcard(recovery.UNACKED_INDEX) == 0


def test_a_job_a_live_worker_is_running_is_left_alone(r):
    _taken(r, "t1", "track-1", "track", ago=600)
    assert recovery.hand_back(r, running={"track-1"}, now=NOW) == []
    assert r.llen("track") == 0 and r.hexists(recovery.UNACKED, "t1")


def test_a_job_taken_moments_ago_is_left_alone(r):
    _taken(r, "t1", "check-1", "check", ago=10)
    assert recovery.hand_back(r, running=set(), now=NOW) == []


def test_builds_and_pushes_keep_the_long_timeout(r):
    _taken(r, "t1", "build-1", "build", ago=6 * 3600)
    assert recovery.hand_back(r, running=set(), now=NOW) == []
    assert r.hexists(recovery.UNACKED, "t1")


def test_nothing_is_handed_back_when_the_workers_cannot_be_asked(r, monkeypatch):
    _taken(r, "t1", "check-1", "check", ago=600)
    monkeypatch.setattr(recovery, "_running_task_ids", lambda app: None)
    assert recovery.sweep_once(object(), r) == []
    assert r.hexists(recovery.UNACKED, "t1")


def test_only_one_worker_sweeps_at_a_time(r, monkeypatch):
    _taken(r, "t1", "check-1", "check", ago=600)
    monkeypatch.setattr(recovery, "_running_task_ids", lambda app: set())
    r.set(recovery.LOCK, "1")
    assert recovery.sweep_once(object(), r) == [], "another worker holds the sweep"
    r.delete(recovery.LOCK)
    assert recovery.sweep_once(object(), r) == ["check-1"]
