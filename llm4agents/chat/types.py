from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TypedDict


class ChatMessage(TypedDict, total=False):
    role: str
    content: str | None
    tool_calls: list[Any] | None
    tool_call_id: str | None


@dataclass(frozen=True)
class ChatResponse:
    id: str
    choices: tuple[dict[str, Any], ...]
    usage: dict[str, int]
    model: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatResponse:
        return cls(
            id=d["id"],
            choices=tuple(d.get("choices", [])),
            usage=d.get("usage") or {},
            model=d.get("model", ""),
        )


@dataclass(frozen=True)
class StreamChunk:
    id: str
    choices: tuple[dict[str, Any], ...]
    usage: dict[str, int] | None
    model: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StreamChunk:
        return cls(
            id=d["id"],
            choices=tuple(d.get("choices", [])),
            usage=d.get("usage"),
            model=d.get("model"),
        )
