from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Any, Callable
from llm4agents.transport.http import HttpTransport
from llm4agents.chat.types import ChatResponse, FinalUsage, StreamChunk


class ChatCompletions:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def create(
        self,
        params: dict[str, Any],
        *,
        on_final_usage: Callable[[FinalUsage], None] | None = None,
    ) -> ChatResponse | AsyncIterator[StreamChunk]:
        if params.get("stream"):
            raw_stream = await self._http.post_stream("/v1/chat/completions", params)
            return self._wrap_stream(raw_stream, on_final_usage)
        data = await self._http.post("/v1/chat/completions", params)
        return ChatResponse.from_dict(data)

    async def _wrap_stream(
        self,
        raw: AsyncIterator[Any],
        on_final_usage: Callable[[FinalUsage], None] | None,
    ) -> AsyncIterator[StreamChunk]:
        last_usage: dict[str, int] | None = None
        async for chunk in raw:
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
