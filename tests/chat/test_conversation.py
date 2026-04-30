import pytest
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock
from llm4agents.transport.http import HttpTransport
from llm4agents.chat.conversation import Conversation
from llm4agents.chat.types import ResponseMeta
from llm4agents.errors import LLM4AgentsError
from llm4agents.tools.types import McpToolResult, McpTextContent


@pytest.fixture
def http():
    return HttpTransport("https://api.example.com", "test-key", 5.0)


@respx.mock
async def test_say_no_tools(http):
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "resp-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!", "tool_calls": None, "tool_call_id": None}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            "model": "gpt-4o",
        })
    )
    conv = Conversation(http, {"model": "gpt-4o"})
    result = await conv.say("Hi")
    assert result.content == "Hello!"
    assert result.tool_calls == []
    assert result.usage["prompt_tokens"] == 10
    assert len(conv.messages) == 2  # user + assistant


@respx.mock
async def test_say_with_tool_call(http):
    tool_response_1 = {
        "id": "resp-1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc-1", "type": "function", "function": {"name": "scrape_url", "arguments": '{"url":"https://example.com"}'}}],
                "tool_call_id": None,
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        "model": "gpt-4o",
    }
    tool_response_2 = {
        "id": "resp-2",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done!", "tool_calls": None, "tool_call_id": None}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 5},
        "model": "gpt-4o",
    }

    mock_tools = MagicMock()
    mock_tools.definitions = [{"name": "scrape_url", "description": "Scrape", "inputSchema": {}}]
    mock_tools.call = AsyncMock(return_value=McpToolResult(
        content=(McpTextContent(type="text", text="<html>content</html>"),),
        text="<html>content</html>",
    ))

    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=tool_response_1 if call_count == 1 else tool_response_2)

    respx.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    conv = Conversation(http, {"model": "gpt-4o", "tools": mock_tools})
    result = await conv.say("Scrape this page")
    assert result.content == "Done!"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "scrape_url"
    mock_tools.call.assert_called_once_with("scrape_url", {"url": "https://example.com"})


async def test_messages_readonly(http):
    conv = Conversation(http, {"model": "gpt-4o"})
    msgs = conv.messages
    assert isinstance(msgs, list)
    # It's a copy — mutating it does not affect the conversation
    msgs.append({"role": "user", "content": "x", "tool_calls": None, "tool_call_id": None})
    assert len(conv.messages) == 0


async def test_clear(http):
    conv = Conversation(http, {"model": "gpt-4o"})
    conv._history.append({"role": "user", "content": "hi", "tool_calls": None, "tool_call_id": None})
    conv.clear()
    assert conv.messages == []


async def test_fork(http):
    conv = Conversation(http, {"model": "gpt-4o"})
    conv._history.append({"role": "user", "content": "hi", "tool_calls": None, "tool_call_id": None})
    fork = conv.fork()
    assert fork.messages == conv.messages
    fork._history.append({"role": "assistant", "content": "yo", "tool_calls": None, "tool_call_id": None})
    assert len(conv.messages) == 1  # original unchanged


async def test_tool_loop_limit(http):
    conv = Conversation(http, {"model": "gpt-4o", "max_tool_rounds": 2})
    # Inject state as if we've already done max_tool_rounds tool iterations
    conv._tool_rounds = 2
    with pytest.raises(LLM4AgentsError) as exc_info:
        conv._check_tool_limit()
    assert exc_info.value.code == "tool_loop_limit"


@respx.mock
async def test_on_round_meta_called(http):
    """on_round_meta callback receives a ResponseMeta after say()."""
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!", "tool_calls": None, "tool_call_id": None}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                "model": "gpt-4o",
            },
            headers={
                "x-request-id": "req-abc",
                "x-model-used": "gpt-4o",
                "x-cost-usd-cents": "5",
                "x-balance-remaining-cents": "1000",
                "x-tokens-input": "10",
                "x-tokens-output": "3",
            },
        )
    )

    received_meta: list[ResponseMeta] = []

    conv = Conversation(http, {"model": "gpt-4o", "on_round_meta": received_meta.append})
    await conv.say("Hello")

    assert len(received_meta) == 1
    meta = received_meta[0]
    assert meta.request_id == "req-abc"
    assert meta.model_used == "gpt-4o"
    assert meta.cost_usd_cents == 5
    assert meta.balance_remaining_cents == 1000
    assert meta.tokens_input == 10
    assert meta.tokens_output == 3


@respx.mock
async def test_on_tools_ignored_say(http):
    """on_tools_ignored is called when model returns no tool_calls on round 0."""
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "resp-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Sure!", "tool_calls": None, "tool_call_id": None}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            "model": "gpt-4o",
        })
    )

    ignored_models: list[str] = []
    mock_tools = MagicMock()
    mock_tools.definitions = [{"name": "my_tool", "description": "A tool", "inputSchema": {}}]

    conv = Conversation(
        http,
        {
            "model": "gpt-4o",
            "tools": mock_tools,
            "on_tools_ignored": ignored_models.append,
        },
    )
    await conv.say("Do something")

    assert ignored_models == ["gpt-4o"]


@respx.mock
async def test_stream_yields_meta_event(http):
    """stream() yields {"type": "meta"} events with ResponseMeta."""
    sse_body = (
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )

    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse_body.encode(),
            headers={
                "content-type": "text/event-stream",
                "x-request-id": "stream-req-1",
                "x-model-used": "gpt-4o",
                "x-cost-usd-cents": "2",
            },
        )
    )

    conv = Conversation(http, {"model": "gpt-4o"})
    events = []
    async for event in conv.stream("Hi"):
        events.append(event)

    meta_events = [e for e in events if e["type"] == "meta"]
    assert len(meta_events) == 1
    meta = meta_events[0]["meta"]
    assert isinstance(meta, ResponseMeta)
    assert meta.request_id == "stream-req-1"
    assert meta.model_used == "gpt-4o"
    assert meta.cost_usd_cents == 2

    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1


@respx.mock
async def test_stream_on_tools_ignored(http):
    """on_tools_ignored is called when stream() round 0 has no tool_calls."""
    sse_body = (
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Sure"},"finish_reason":null}]}\n\n'
        'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )

    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
        )
    )

    ignored_models: list[str] = []
    mock_tools = MagicMock()
    mock_tools.definitions = [{"name": "my_tool", "description": "A tool", "inputSchema": {}}]

    conv = Conversation(
        http,
        {
            "model": "gpt-4o",
            "tools": mock_tools,
            "on_tools_ignored": ignored_models.append,
        },
    )
    events = []
    async for event in conv.stream("Do something"):
        events.append(event)

    assert ignored_models == ["gpt-4o"]
