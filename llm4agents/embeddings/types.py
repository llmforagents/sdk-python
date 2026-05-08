from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class EmbeddingItem:
    embedding: Sequence[float] | str
    index: int
    object: str = "embedding"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmbeddingItem":
        return cls(
            embedding=d["embedding"],
            index=int(d.get("index", 0)),
            object=d.get("object", "embedding"),
        )


@dataclass(frozen=True)
class EmbeddingsUsage:
    prompt_tokens: int
    total_tokens: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmbeddingsUsage":
        return cls(
            prompt_tokens=int(d.get("prompt_tokens", 0) or 0),
            total_tokens=int(d.get("total_tokens", 0) or 0),
        )


@dataclass(frozen=True)
class EmbeddingsResponse:
    data: list[EmbeddingItem]
    model: str
    usage: EmbeddingsUsage
    object: str = "list"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmbeddingsResponse":
        return cls(
            data=[EmbeddingItem.from_dict(x) for x in d.get("data", [])],
            model=str(d.get("model", "")),
            usage=EmbeddingsUsage.from_dict(d.get("usage") or {}),
            object=d.get("object", "list"),
        )
