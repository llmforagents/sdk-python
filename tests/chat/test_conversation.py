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


# ---------------------------------------------------------------------------
# Fix 1 (TS v2.2.0): prompt-mode tool fallback
# ---------------------------------------------------------------------------


@respx.mock
async def test_say_prompt_fallback_executes_parsed_tool_call(http):
    """When a model ignores native tools, the SDK retries with tools in the
    system prompt and parses ``<tool_call>`` blocks from the response."""
    # Round 0: no tool_calls
    round0 = {
        "id": "r0",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "I cannot do that.", "tool_calls": None, "tool_call_id": None},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        "model": "test-model",
    }
    # Fallback round: model emits a tool_call block
    fallback_round = {
        "id": "rfb",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '<tool_call>{"name":"echo","arguments":{"x":1}}</tool_call>',
                "tool_calls": None,
                "tool_call_id": None,
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        "model": "test-model",
    }
    # Round 2: final answer after tool result
    final_round = {
        "id": "rf",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Done!", "tool_calls": None, "tool_call_id": None},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 3},
        "model": "test-model",
    }

    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, json=round0)
        if call_count == 2:
            return httpx.Response(200, json=fallback_round)
        return httpx.Response(200, json=final_round)

    respx.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    mock_tools = MagicMock()
    mock_tools.definitions = [{"name": "echo", "description": "Echo input", "inputSchema": {}}]
    mock_tools.call = AsyncMock(return_value=McpToolResult(
        content=(McpTextContent(type="text", text="echoed"),),
        text="echoed",
    ))

    conv = Conversation(
        http,
        {
            "model": "test-model",
            "tools": mock_tools,
            "enable_prompt_tool_fallback": True,
        },
    )
    result = await conv.say("Use the tool")

    assert call_count == 3  # round0 + fallback + final
    assert result.content == "Done!"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "echo"
    mock_tools.call.assert_called_once_with("echo", {"x": 1})
    # Usage accumulated across all three rounds
    assert result.usage["prompt_tokens"] == 40
    assert result.usage["completion_tokens"] == 13
    assert result.usage["total_tokens"] == 53


@respx.mock
async def test_say_prompt_fallback_no_blocks_returns_text(http):
    """If the fallback round still has no <tool_call> blocks, return its text."""
    round0 = {
        "id": "r0",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ignored", "tool_calls": None, "tool_call_id": None},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        "model": "test-model",
    }
    fallback_round = {
        "id": "rfb",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "No tool needed.", "tool_calls": None, "tool_call_id": None},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 4},
        "model": "test-model",
    }
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=round0 if call_count == 1 else fallback_round)

    respx.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    mock_tools = MagicMock()
    mock_tools.definitions = [{"name": "echo", "description": "Echo input", "inputSchema": {}}]

    conv = Conversation(
        http,
        {
            "model": "test-model",
            "tools": mock_tools,
            "enable_prompt_tool_fallback": True,
        },
    )
    result = await conv.say("hi")

    assert call_count == 2
    assert result.content == "No tool needed."
    assert result.tool_calls == []


@respx.mock
async def test_stream_prompt_fallback_yields_fallback_event(http):
    """Stream emits a ``fallback`` event before invoking the prompt-mode call."""
    sse_body = (
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"ignored"},"finish_reason":null}]}\n\n'
        'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    fallback_round = {
        "id": "rfb",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Direct answer.", "tool_calls": None, "tool_call_id": None},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 6, "completion_tokens": 3},
        "model": "test-model",
    }

    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                200,
                content=sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=fallback_round)

    respx.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    mock_tools = MagicMock()
    mock_tools.definitions = [{"name": "echo", "description": "Echo input", "inputSchema": {}}]

    conv = Conversation(
        http,
        {
            "model": "test-model",
            "tools": mock_tools,
            "enable_prompt_tool_fallback": True,
        },
    )
    events = []
    async for event in conv.stream("hi"):
        events.append(event)

    fallback_events = [e for e in events if e["type"] == "fallback"]
    assert len(fallback_events) == 1
    assert fallback_events[0]["reason"] == "tools_ignored"
    assert fallback_events[0]["model"] == "test-model"

    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["response"]["content"] == "Direct answer."


@respx.mock
async def test_stream_prompt_fallback_executes_parsed_tool_call(http):
    """Stream fallback parses <tool_call> blocks and executes them."""
    sse_body = (
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hmm"},"finish_reason":null}]}\n\n'
        'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    fallback_round = {
        "id": "rfb",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '<tool_call>{"name":"echo","arguments":{}}</tool_call>',
                "tool_calls": None,
                "tool_call_id": None,
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 6, "completion_tokens": 3},
        "model": "test-model",
    }
    sse_final = (
        'data: {"id":"c2","choices":[{"index":0,"delta":{"content":"All set."},"finish_reason":null}]}\n\n'
        'data: {"id":"c2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":4}}\n\n'
        "data: [DONE]\n\n"
    )

    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                200,
                content=sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        if call_count == 2:
            return httpx.Response(200, json=fallback_round)
        return httpx.Response(
            200,
            content=sse_final.encode(),
            headers={"content-type": "text/event-stream"},
        )

    respx.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    mock_tools = MagicMock()
    mock_tools.definitions = [{"name": "echo", "description": "Echo input", "inputSchema": {}}]
    mock_tools.call = AsyncMock(return_value=McpToolResult(
        content=(McpTextContent(type="text", text="ok"),),
        text="ok",
    ))

    conv = Conversation(
        http,
        {
            "model": "test-model",
            "tools": mock_tools,
            "enable_prompt_tool_fallback": True,
        },
    )
    events = []
    async for event in conv.stream("hi"):
        events.append(event)

    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0]["name"] == "echo"
    mock_tools.call.assert_called_once_with("echo", {})

    done = [e for e in events if e["type"] == "done"]
    assert done[0]["response"]["content"] == "All set."


# ---------------------------------------------------------------------------
# Fix 4 (BUG-03): reasoning_tokens propagation in usage + ResponseMeta header
# ---------------------------------------------------------------------------


@respx.mock
async def test_say_propagates_reasoning_tokens(http):
    """Reasoning tokens accumulate across rounds and surface in ResponseMeta."""
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "r1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Done", "tool_calls": None, "tool_call_id": None},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 7},
                "model": "gpt-5",
            },
            headers={"x-tokens-reasoning": "7"},
        )
    )

    received_meta: list[ResponseMeta] = []
    conv = Conversation(http, {"model": "gpt-5", "on_round_meta": received_meta.append})
    result = await conv.say("hi")

    assert result.usage["reasoning_tokens"] == 7
    assert result.usage["total_tokens"] == 15
    assert received_meta[0].tokens_reasoning == 7


# ---------------------------------------------------------------------------
# Fix 7 (BUG-08): assistant.content normalization (no None in history)
# ---------------------------------------------------------------------------


@respx.mock
async def test_say_normalizes_null_content_to_empty_string(http):
    """OpenAI/Gemini return content:None for tool-only messages — must coerce to ''."""
    tool_response = {
        "id": "r1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,  # ← null
                "tool_calls": [{"id": "tc-1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
                "tool_call_id": None,
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        "model": "gpt-4o",
    }
    final_response = {
        "id": "r2",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Done", "tool_calls": None, "tool_call_id": None},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 8, "completion_tokens": 1},
        "model": "gpt-4o",
    }

    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=tool_response if call_count == 1 else final_response)

    respx.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    mock_tools = MagicMock()
    mock_tools.definitions = [{"name": "echo", "description": "Echo", "inputSchema": {}}]
    mock_tools.call = AsyncMock(return_value=McpToolResult(
        content=(McpTextContent(type="text", text="ok"),),
        text="ok",
    ))

    conv = Conversation(http, {"model": "gpt-4o", "tools": mock_tools})
    await conv.say("Use the tool")

    # The first assistant message should have content "" not None
    assistant_msgs = [m for m in conv.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) >= 1
    # Tool-only response (round 0) had content=None → must be normalized
    assert assistant_msgs[0]["content"] == ""
    assert assistant_msgs[0]["content"] is not None


@respx.mock
async def test_stream_assistant_content_is_string_not_none(http):
    """Stream() must push content as '' (not None) when no text was streamed."""
    # Streaming round with only a tool_call — no content delta
    sse_body = (
        'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"tc-1","function":{"name":"echo","arguments":"{}"}}]},"finish_reason":null}]}\n\n'
        'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    sse_final = (
        'data: {"id":"c2","choices":[{"index":0,"delta":{"content":"All set."},"finish_reason":null}]}\n\n'
        'data: {"id":"c2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    )
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        body = sse_body if call_count == 1 else sse_final
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    respx.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    mock_tools = MagicMock()
    mock_tools.definitions = [{"name": "echo", "description": "Echo", "inputSchema": {}}]
    mock_tools.call = AsyncMock(return_value=McpToolResult(
        content=(McpTextContent(type="text", text="ok"),),
        text="ok",
    ))

    conv = Conversation(http, {"model": "gpt-4o", "tools": mock_tools})
    async for _ in conv.stream("Use the tool"):
        pass

    assistant_msgs = [m for m in conv.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) >= 1
    # First assistant message had no text delta → content must be ""
    assert assistant_msgs[0]["content"] == ""
    assert assistant_msgs[0]["content"] is not None
