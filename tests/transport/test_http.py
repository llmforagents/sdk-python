import pytest
import respx
import httpx
from llm4agents.transport.http import HttpTransport
from llm4agents.errors import LLM4AgentsError


@pytest.fixture
def transport():
    return HttpTransport("https://api.example.com", "test-key", 5.0)


@respx.mock
async def test_get_success(transport):
    respx.get("https://api.example.com/api/v1/models/").mock(
        return_value=httpx.Response(200, json={"models": [{"slug": "gpt-4o"}]})
    )
    result = await transport.get("/api/v1/models/")
    assert result["models"][0]["slug"] == "gpt-4o"


@respx.mock
async def test_post_success(transport):
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "resp-1", "choices": []})
    )
    result = await transport.post("/v1/chat/completions", {"model": "gpt-4o", "messages": []})
    assert result["id"] == "resp-1"


@respx.mock
async def test_post_401_raises(transport):
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Unauthorized"}})
    )
    with pytest.raises(LLM4AgentsError) as exc_info:
        await transport.post("/v1/chat/completions", {})
    assert exc_info.value.code == "auth_error"
    assert exc_info.value.status_code == 401
