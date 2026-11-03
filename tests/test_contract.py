"""What this project and the app agree on, pinned.

The app's repository (RedMindRD/data-collection-platform) pins the same values
in backend/tests/test_contract.py. Change one here without the other and one
of the two suites fails — which is the point: the two projects share no code,
so this is what keeps them in step. See CONTRACT.md.
"""

from dfpipeline import PROTOCOL_VERSION
from dfpipeline import compute as computes
from dfpipeline.worker import DB_QUEUE, celery_app


def test_the_protocol_version():
    assert PROTOCOL_VERSION == 4


def test_the_computes_this_pipeline_offers():
    assert sorted(computes._COMPUTE) == [
        "dataset.convert", "dataset.publish", "export.build", "retarget.solve", "track.hand", "validate.judge",
    ]


def test_the_task_names_on_the_wire():
    assert "pipeline.compute" in celery_app.tasks
    assert DB_QUEUE == "db"


def test_the_queue_settings_match_the_app():
    conf = celery_app.conf
    assert conf.task_acks_late is True
    assert conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1
    assert conf.task_serializer == "json"
    assert conf.broker_transport_options == {"visibility_timeout": 9 * 3600}


def test_the_storage_paths():
    from dfpipeline.retarget import service as rs

    rs.settings.retarget_pipeline_version  # noqa: B018  (the same setting on both servers)
    assert rs.bundle_path("v1") == f"tracks/{rs.settings.retarget_pipeline_version}/v1.zip"
    assert not hasattr(rs, "overlay_path")  # the tracked video is gone
    assert rs.store_path("r1", "v1") == f"retargets/{rs.settings.retarget_pipeline_version}/r1/v1.json.gz"


def test_the_shared_hours_and_heartbeat_keys():
    from dfpipeline import heartbeat

    assert heartbeat.HOURS_KEY == "pipeline:hours"
    assert heartbeat.KEY == "pipeline:heartbeat:{node}"
    assert heartbeat.TTL_SEC == 90
