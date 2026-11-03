"""The hours come from the app; the pipeline checks in (dfpipeline/heartbeat.py)."""

import json

import fakeredis

from dfpipeline import heartbeat
from dfpipeline.config import settings
from dfpipeline.redis import use_redis


def test_the_hours_set_in_the_app_win_over_the_env(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    use_redis(client)
    try:
        monkeypatch.setattr(settings, "pipeline_hours", "22:00-09:00")
        assert heartbeat.working_hours()[0] == "22:00-09:00"  # nothing set: the env
        client.set("pipeline:hours", json.dumps({"hours": "", "tz": "Asia/Kolkata"}))
        assert heartbeat.working_hours() == ("", "Asia/Kolkata")  # "always", from the app
    finally:
        use_redis(None)


def test_a_beat_says_who_and_whether_asleep(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    use_redis(client)
    try:
        monkeypatch.setattr(settings, "node_name", "dgx")
        heartbeat.beat(True, "22:00-09:00", "Asia/Kolkata")
        beat = json.loads(client.get("pipeline:heartbeat:dgx"))
        assert beat["node"] == "dgx" and beat["asleep"] is True and beat["hours"] == "22:00-09:00"
        assert 0 < client.ttl("pipeline:heartbeat:dgx") <= heartbeat.TTL_SEC
    finally:
        use_redis(None)
