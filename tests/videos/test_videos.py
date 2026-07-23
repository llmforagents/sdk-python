import pytest
import respx
import httpx
from llm4agents.transport.http import HttpTransport
from llm4agents.videos.videos import Videos
from llm4agents.errors import LLM4AgentsError


@pytest.fixture
def transport():
    return HttpTransport("https://api.example.com", "test-key", 5.0)


@pytest.fixture
def videos(transport):
    return Videos(transport)


@respx.mock
async def test_create_posts_to_v1_videos_and_returns_accepted_shape(videos):
    respx.post("https://api.example.com/v1/videos").mock(
        return_value=httpx.Response(
            202,
            json={
                "id": "job_1",
                "status": "pending",
                "polling_url": "/v1/videos/job_1",
                "charged_usd_cents": 250,
            },
            headers={"x-request-id": "req_video_1"},
        )
    )
    result = await videos.create(
        prompt="A cat riding a skateboard",
        model="kling-2.5",
        duration=5,
        resolution="720p",
    )
    assert result.id == "job_1"
    assert result.status == "pending"
    assert result.polling_url == "/v1/videos/job_1"
    assert result.charged_usd_cents == 250

    sent = respx.calls.last.request
    import json as _json
    sent_body = _json.loads(sent.content)
    assert sent_body == {
        "prompt": "A cat riding a skateboard",
        "model": "kling-2.5",
        "duration": 5,
        "resolution": "720p",
    }


@respx.mock
async def test_create_omits_none_fields(videos):
    respx.post("https://api.example.com/v1/videos").mock(
        return_value=httpx.Response(
            202,
            json={
                "id": "job_2",
                "status": "pending",
                "polling_url": "/v1/videos/job_2",
                "charged_usd_cents": 100,
            },
        )
    )
    await videos.create(prompt="minimal prompt")
    sent = respx.calls.last.request
    import json as _json
    assert _json.loads(sent.content) == {"prompt": "minimal prompt"}


@respx.mock
async def test_create_402_raises_insufficient_balance(videos):
    respx.post("https://api.example.com/v1/videos").mock(
        return_value=httpx.Response(
            402,
            json={"error": {"code": "insufficient_balance", "message": "Not enough balance"}},
            headers={"x-request-id": "req_402"},
        )
    )
    with pytest.raises(LLM4AgentsError) as exc_info:
        await videos.create(prompt="A cat riding a skateboard")
    assert exc_info.value.code == "insufficient_balance"
    assert exc_info.value.status_code == 402
    assert exc_info.value.request_id == "req_402"


@respx.mock
async def test_get_returns_status_shape(videos):
    respx.get("https://api.example.com/v1/videos/job_1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "job_1",
                "status": "completed",
                "video_url": "https://cdn.example.com/job_1.mp4",
                "charged_usd_cents": 250,
            },
        )
    )
    result = await videos.get("job_1")
    assert result.status == "completed"
    assert result.video_url == "https://cdn.example.com/job_1.mp4"
    assert result.charged_usd_cents == 250
    assert result.error is None
    assert result.refunded is None


@respx.mock
async def test_get_encodes_job_id_in_url(videos):
    route = respx.get("https://api.example.com/v1/videos/job%2F1").mock(
        return_value=httpx.Response(
            200,
            json={"id": "job/1", "status": "pending", "charged_usd_cents": 0},
        )
    )
    await videos.get("job/1")
    assert route.called


@respx.mock
async def test_get_surfaces_failed_job_with_error_and_refund_flag(videos):
    respx.get("https://api.example.com/v1/videos/job_2").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "job_2",
                "status": "failed",
                "error": "provider timeout",
                "refunded": True,
                "charged_usd_cents": 0,
            },
        )
    )
    result = await videos.get("job_2")
    assert result.status == "failed"
    assert result.error == "provider timeout"
    assert result.refunded is True


@respx.mock
async def test_content_returns_bytes_and_content_type(videos):
    video_bytes = b"\x00\x00\x00\x18ftypmp42fake-mp4-bytes"
    respx.get("https://api.example.com/v1/videos/job_1/content").mock(
        return_value=httpx.Response(
            200,
            content=video_bytes,
            headers={"content-type": "video/mp4", "x-request-id": "req_video_3"},
        )
    )
    result = await videos.content("job_1")
    assert result.data == video_bytes
    assert result.content_type == "video/mp4"
    assert result.request_id == "req_video_3"


@respx.mock
async def test_content_defaults_content_type_when_missing(videos):
    respx.get("https://api.example.com/v1/videos/job_1/content").mock(
        return_value=httpx.Response(200, content=b"raw-bytes")
    )
    result = await videos.content("job_1")
    assert result.content_type == "video/mp4"
    assert result.request_id is None


@respx.mock
async def test_content_402_raises_insufficient_balance(videos):
    respx.get("https://api.example.com/v1/videos/job_1/content").mock(
        return_value=httpx.Response(
            402,
            json={"error": {"code": "insufficient_balance", "message": "Not enough balance"}},
            headers={"x-request-id": "req_402"},
        )
    )
    with pytest.raises(LLM4AgentsError) as exc_info:
        await videos.content("job_1")
    assert exc_info.value.code == "insufficient_balance"
    assert exc_info.value.status_code == 402
    assert exc_info.value.request_id == "req_402"
