from __future__ import annotations
from typing import Any
from llm4agents.transport.http import HttpTransport
from llm4agents.images.types import GeneratedImage, ImagesGenerateResponse


class Images:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        n: int | None = None,
        **kwargs: Any,
    ) -> ImagesGenerateResponse:
        payload: dict[str, object] = {
            "prompt": prompt,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        if model is not None:
            payload["model"] = model
        if n is not None:
            payload["n"] = n
        data = await self._http.post("/v1/images/generations", payload)
        usage = data.get("usage") or {}
        return ImagesGenerateResponse(
            created=data.get("created"),
            data=[
                GeneratedImage(b64_json=d["b64_json"], media_type=d.get("media_type"))
                for d in data.get("data", [])
            ],
            cost_usd=usage.get("cost"),
        )
