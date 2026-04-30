from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TypedDict
import httpx


class ChatMessage(TypedDict, total=False):
    role: str
    content: str | None
    tool_calls: list[Any] | None
    tool_call_id: str | None


@dataclass(frozen=True)
class ResponseMeta:
    request_id: str | None
    model_used: str | None
    cost_usd_cents: int | None
    balance_remaining_cents: int | None
    tokens_input: int | None
    tokens_output: int | None

    @classmethod
    def from_headers(cls, headers: httpx.Headers) -> ResponseMeta:
        def parse_int(name: str) -> int | None:
            val = headers.get(name)
            if val is None:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        return cls(
            request_id=headers.get("x-request-id"),
            model_used=headers.get("x-model-used"),
            cost_usd_cents=parse_int("x-cost-usd-cents"),
            balance_remaining_cents=parse_int("x-balance-remaining-cents"),
            tokens_input=parse_int("x-tokens-input"),
            tokens_output=parse_int("x-tokens-output"),
        )


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
