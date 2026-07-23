from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratedImage:
    b64_json: str
    media_type: str | None


@dataclass(frozen=True)
class ImagesGenerateResponse:
    created: int | None
    data: list[GeneratedImage]
    cost_usd: float | None
