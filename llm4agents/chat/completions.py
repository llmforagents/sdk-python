from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Any, Callable
from llm4agents.transport.http import HttpTransport
from llm4agents.chat.types import ChatResponse, FinalUsage, StreamChunk
from llm4agents.x402.types import X402Receipt


class ChatCompletions:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def create(
        self,
        params: dict[str, Any],
        *,
        on_final_usage: Callable[[FinalUsage], None] | None = None,
        on_x402_receipt: Callable[[X402Receipt], None] | None = None,
    ) -> ChatResponse | AsyncIterator[StreamChunk]:
        if params.get("stream"):
            raw_stream = await self._http.post_stream("/v1/chat/completions", params)
            return self._wrap_stream(raw_stream, on_final_usage, on_x402_receipt)
        data = await self._http.post("/v1/chat/completions", params)
        return ChatResponse.from_dict(data)

    async def _wrap_stream(
        self,
        raw: AsyncIterator[Any],
        on_final_usage: Callable[[FinalUsage], None] | None,
        on_x402_receipt: Callable[[X402Receipt], None] | None,
    ) -> AsyncIterator[StreamChunk]:
        last_usage: dict[str, int] | None = None
        async for chunk in raw:
            # Trailing typed SSE event (e.g. x402-receipt). The transport
            # tags these as {"_event": "<name>", "data": {...}} so they
            # don't get misread as chat chunks.
            event_name = chunk.get("_event") if isinstance(chunk, dict) else None
            if event_name == "x402-receipt":
                data = chunk.get("data") or {}
                if on_x402_receipt is not None and _is_well_formed_receipt(data):
                    on_x402_receipt(
                        X402Receipt(
                            transaction=data["transaction"],
                            network=data["network"],
                            amount=data["amount"],
                            payer=data["payer"],
                        )
                    )
                continue
            if event_name is not None:
                # Unknown typed event — skip (forward-compat).
                continue
            usage = chunk.get("usage")
            if usage:
                last_usage = usage
            yield StreamChunk.from_dict(chunk)
        if last_usage is not None and on_final_usage is not None:
            prompt = int(last_usage.get("prompt_tokens", 0) or 0)
            completion = int(last_usage.get("completion_tokens", 0) or 0)
            reasoning_raw = last_usage.get("reasoning_tokens")
            reasoning = int(reasoning_raw) if reasoning_raw is not None else None
            on_final_usage(
                FinalUsage(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=prompt + completion,
                    reasoning_tokens=reasoning,
                )
            )


def _is_well_formed_receipt(data: dict[str, Any]) -> bool:
    return all(
        isinstance(data.get(k), str)
        for k in ("transaction", "network", "amount", "payer")
    )
