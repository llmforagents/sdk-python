import pytest
import respx
import httpx
from llm4agents.transport.http import HttpTransport
from llm4agents.chat.completions import ChatCompletions
from llm4agents.chat.types import ChatResponse, StreamChunk


@pytest.fixture
def completions():
    return ChatCompletions(HttpTransport("https://api.example.com", "test-key", 5.0))


@respx.mock
async def test_create_non_streaming(completions):
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!", "tool_calls": None, "tool_call_id": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4o",
        })
    )
    result = await completions.create({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi", "tool_calls": None, "tool_call_id": None}],
    })
    assert isinstance(result, ChatResponse)
    assert result.id == "chatcmpl-1"
    assert result.choices[0]["message"]["content"] == "Hello!"
    assert result.usage["prompt_tokens"] == 10


@respx.mock
async def test_create_streaming(completions):
    sse_body = (
        'data: {"id":"stream-1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi","tool_calls":null},"finish_reason":null}],"usage":null,"model":"gpt-4o"}\n\n'
        'data: {"id":"stream-1","choices":[{"index":0,"delta":{"role":null,"content":"!","tool_calls":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2},"model":"gpt-4o"}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})
    )
    chunks = []
    async for chunk in await completions.create({"model": "gpt-4o", "messages": [], "stream": True}):
        chunks.append(chunk)
    assert len(chunks) == 2
    assert isinstance(chunks[0], StreamChunk)
    assert chunks[0].choices[0]["delta"]["content"] == "Hi"
