import os
import tempfile

# No app server, no Redis, no Hub: every test that needs one stands it in.
os.environ.setdefault("ENV_FILE", "/nonexistent.env")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("STAGING_DIR", tempfile.mkdtemp())
os.environ.setdefault("EXPORT_DATA_DIR", tempfile.mkdtemp())

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_redis():
    """No test reaches a real Redis; one that needs it stands one in."""
    from dfpipeline.redis import use_redis

    use_redis(None)
    yield
    use_redis(None)
