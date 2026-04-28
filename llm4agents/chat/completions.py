from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Any
from llm4agents.transport.http import HttpTransport
from llm4agents.chat.types import ChatResponse, StreamChunk


class ChatCompletions:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def create(
        self, params: dict[str, Any]
    ) -> ChatResponse | AsyncIterator[StreamChunk]:
        if params.get("stream"):
            raw_stream = await self._http.post_stream("/v1/chat/completions", params)
            return self._wrap_stream(raw_stream)
        data = await self._http.post("/v1/chat/completions", params)
        return ChatResponse.from_dict(data)

    async def _wrap_stream(
        self, raw: AsyncIterator[Any]
    ) -> AsyncIterator[StreamChunk]:
        async for chunk in raw:
            yield StreamChunk.from_dict(chunk)
