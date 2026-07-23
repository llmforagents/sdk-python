import pytest
import respx
import httpx
from llm4agents.transport.http import HttpTransport
from llm4agents.images.images import Images
from llm4agents.errors import LLM4AgentsError


@pytest.fixture
def transport():
    return HttpTransport("https://api.example.com", "test-key", 5.0)


@pytest.fixture
def images(transport):
    return Images(transport)


@respx.mock
async def test_generate_posts_to_v1_images_generations_and_returns_parsed_shape(images):
    respx.post("https://api.example.com/v1/images/generations").mock(
        return_value=httpx.Response(
            200,
            json={
                "created": 1700000000,
                "data": [
                    {"b64_json": "aGVsbG8=", "media_type": "image/png"},
                ],
                "usage": {"cost": 0.04},
            },
        )
    )
    result = await images.generate(
        prompt="A robot writing code, studio lighting",
        model="x-ai/grok-image-1.0",
        n=1,
        resolution="1K",
    )
    assert result.created == 1700000000
    assert result.cost_usd == 0.04
    assert len(result.data) == 1
    assert result.data[0].b64_json == "aGVsbG8="
    assert result.data[0].media_type == "image/png"

    sent = respx.calls.last.request
    import json as _json
    sent_body = _json.loads(sent.content)
    assert sent_body == {
        "prompt": "A robot writing code, studio lighting",
        "resolution": "1K",
        "model": "x-ai/grok-image-1.0",
        "n": 1,
    }


@respx.mock
async def test_generate_omits_none_fields(images):
    respx.post("https://api.example.com/v1/images/generations").mock(
        return_value=httpx.Response(
            200,
            json={
                "created": 1700000001,
                "data": [{"b64_json": "aGk=", "media_type": None}],
                "usage": {"cost": 0.02},
            },
        )
    )
    await images.generate(prompt="minimal prompt")
    sent = respx.calls.last.request
    import json as _json
    assert _json.loads(sent.content) == {"prompt": "minimal prompt"}


@respx.mock
async def test_generate_defaults_missing_created_and_usage(images):
    respx.post("https://api.example.com/v1/images/generations").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"b64_json": "aGk="}]},
        )
    )
    result = await images.generate(prompt="no usage or created")
    assert result.created is None
    assert result.cost_usd is None
    assert result.data[0].media_type is None


@respx.mock
async def test_generate_402_raises_insufficient_balance(images):
    respx.post("https://api.example.com/v1/images/generations").mock(
        return_value=httpx.Response(
            402,
            json={"error": {"code": "insufficient_balance", "message": "Not enough balance"}},
            headers={"x-request-id": "req_402"},
        )
    )
    with pytest.raises(LLM4AgentsError) as exc_info:
        await images.generate(prompt="A robot writing code")
    assert exc_info.value.code == "insufficient_balance"
    assert exc_info.value.status_code == 402
    assert exc_info.value.request_id == "req_402"
