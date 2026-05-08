from __future__ import annotations
from typing import Sequence
from llm4agents.transport.http import HttpTransport
from llm4agents.embeddings.types import EmbeddingsResponse


class Embeddings:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def create(
        self,
        *,
        model: str,
        input: str | Sequence[str],
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingsResponse:
        payload: dict[str, object] = {"model": model, "input": input}
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if user is not None:
            payload["user"] = user
        data = await self._http.post("/v1/embeddings", payload)
        return EmbeddingsResponse.from_dict(data)
