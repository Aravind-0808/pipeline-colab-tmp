"""The model client (dfpipeline/llm.py)."""

from dfpipeline.llm import LlmClient


def test_one_call_is_one_request():
    """The SDK retries a failed call twice by itself, out of the breaker's
    sight: a dead model saw every check three times."""
    client = LlmClient(api_base="http://gw:4000/v1", api_key="k", model="m")
    assert client._client().max_retries == 0
