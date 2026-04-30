from __future__ import annotations
import pytest
import respx
import httpx
from llm4agents.transport.http import HttpTransport
from llm4agents.agents import Agents, AgentRegistration
from llm4agents.errors import LLM4AgentsError


@pytest.fixture
def agents():
    return Agents(HttpTransport("https://api.example.com", "test-key", 5.0))


@respx.mock
async def test_register_success(agents):
    """register() returns AgentRegistration on success."""
    respx.post("https://api.example.com/api/v1/agents/register").mock(
        return_value=httpx.Response(
            200,
            json={
                "uuid": "agent-uuid-123",
                "apiKey": "sk-proxy-abc",
                "name": "my-agent",
                "createdAt": "2026-04-30T00:00:00Z",
                "requestId": "req-reg-1",
                "depositDeadline": "2026-05-30T00:00:00Z",
                "depositRequiredWithinMinutes": 43200,
                "notice": "Fund within 15 minutes or registration will expire.",
            },
        )
    )
    result = await agents.register("my-agent")
    assert isinstance(result, AgentRegistration)
    assert result.uuid == "agent-uuid-123"
    assert result.api_key == "sk-proxy-abc"
    assert result.name == "my-agent"
    assert result.request_id == "req-reg-1"


@respx.mock
async def test_register_rate_limited(agents):
    """register() raises LLM4AgentsError with code 'rate_limited' on 429."""
    respx.post("https://api.example.com/api/v1/agents/register").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "Too many requests", "code": "rate_limited"}},
        )
    )
    with pytest.raises(LLM4AgentsError) as exc_info:
        await agents.register("spammy-agent")
    assert exc_info.value.code == "rate_limited"
    assert exc_info.value.status_code == 429
