"""Verify Conversation.stream() emits the x402_receipt event.

The proxy emits a trailing ``event: x402-receipt\\ndata: {...}`` SSE chunk
AFTER ``data: [DONE]`` in x402 walk-up streaming mode. The transport-layer
SSE parser surfaces this as a typed event marker
(``{"_event": "x402-receipt", "data": {...}}``). The Conversation.stream()
generator must capture it and yield ``{"type": "x402_receipt", ...}``
BEFORE the matching ``done`` event.
"""
from __future__ import annotations

import httpx
import respx

from llm4agents.chat.conversation import Conversation
from llm4agents.transport.http import HttpTransport


def _sse_stream_with_receipt() -> str:
    return "\n".join(
        [
            'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hi"}}]}',
            "",
            'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "",
            "data: [DONE]",
            "",
            "event: x402-receipt",
            'data: {"transaction":"0xdeadbeef","network":"eip155:84532","amount":"10000","payer":"0xpayer000000000000000000000000000000000001"}',
            "",
        ]
    )


def _sse_stream_no_receipt() -> str:
    return "\n".join(
        [
            'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hi"}}]}',
            "",
            'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "",
            "data: [DONE]",
            "",
        ]
    )


@respx.mock
async def test_conversation_stream_yields_x402_receipt_before_done() -> None:
    respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_sse_stream_with_receipt().encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )
    )
    http = HttpTransport("https://api.test.com", "sk-test", 5.0)
    conv = Conversation(http, {"model": "openai/gpt-4o-mini"})

    events: list[dict] = []
    async for ev in conv.stream("hello"):
        events.append(ev)

    types = [e["type"] for e in events]
    assert "text" in types
    assert "x402_receipt" in types
    assert "done" in types

    # Order invariant: x402_receipt MUST come before done
    receipt_idx = types.index("x402_receipt")
    done_idx = types.index("done")
    assert receipt_idx < done_idx, (
        f"x402_receipt must precede done, got order: {types}"
    )

    receipt = next(e for e in events if e["type"] == "x402_receipt")
    assert receipt["transaction"] == "0xdeadbeef"
    assert receipt["network"] == "eip155:84532"
    assert receipt["amount"] == "10000"
    assert receipt["payer"] == "0xpayer000000000000000000000000000000000001"


@respx.mock
async def test_conversation_stream_no_receipt_event_in_bearer_mode() -> None:
    respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_sse_stream_no_receipt().encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )
    )
    http = HttpTransport("https://api.test.com", "sk-test", 5.0)
    conv = Conversation(http, {"model": "openai/gpt-4o-mini"})

    events: list[dict] = []
    async for ev in conv.stream("hello"):
        events.append(ev)

    types = [e["type"] for e in events]
    assert "x402_receipt" not in types
