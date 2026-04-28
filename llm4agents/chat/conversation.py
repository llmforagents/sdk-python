from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Callable
from llm4agents.transport.http import HttpTransport
from llm4agents.chat.types import ChatMessage
from llm4agents.errors import LLM4AgentsError


@dataclass(frozen=True)
class ConversationResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    usage: dict[str, int]


class Conversation:
    def __init__(self, http: HttpTransport, opts: dict[str, Any]) -> None:
        self._http = http
        self._model: str = opts["model"]
        self._system: str | None = opts.get("system")
        self._tools = opts.get("tools")
        self._on_tool_call: Callable[[str, Any], bool] | None = opts.get("on_tool_call")
        self._on_tool_result: Callable[[str, Any], None] | None = opts.get("on_tool_result")
        self._max_tool_rounds: int = opts.get("max_tool_rounds", 10)
        self._history: list[ChatMessage] = list(opts.get("history", []))
        self._tool_rounds: int = 0

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        self._tool_rounds = 0

    def fork(self) -> Conversation:
        opts: dict[str, Any] = {
            "model": self._model,
            "max_tool_rounds": self._max_tool_rounds,
            "history": list(self._history),
        }
        if self._system is not None:
            opts["system"] = self._system
        if self._tools is not None:
            opts["tools"] = self._tools
        if self._on_tool_call is not None:
            opts["on_tool_call"] = self._on_tool_call
        if self._on_tool_result is not None:
            opts["on_tool_result"] = self._on_tool_result
        return Conversation(self._http, opts)

    def _check_tool_limit(self) -> None:
        if self._tool_rounds >= self._max_tool_rounds:
            raise LLM4AgentsError(
                f"Tool loop limit reached ({self._max_tool_rounds})",
                "tool_loop_limit",
                None,
                None,
            )

    async def say(self, message: str) -> ConversationResponse:
        self._history.append(
            {"role": "user", "content": message, "tool_calls": None, "tool_call_id": None}
        )

        all_tool_calls: list[dict[str, Any]] = []
        final_usage: dict[str, int] = {}

        while True:
            params: dict[str, Any] = {
                "model": self._model,
                "messages": self._build_messages(),
            }
            if self._tools is not None:
                params["tools"] = self._tools.definitions

            data = await self._http.post("/v1/chat/completions", params)
            choice = data["choices"][0]
            msg: ChatMessage = choice["message"]
            usage: dict[str, int] = data.get("usage") or {}
            final_usage = usage
            finish_reason: str = choice.get("finish_reason", "stop")

            self._history.append(msg)

            raw_tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []

            if not raw_tool_calls or finish_reason == "stop":
                return ConversationResponse(
                    content=msg.get("content") or "",
                    tool_calls=all_tool_calls,
                    usage=final_usage,
                )

            self._check_tool_limit()
            self._tool_rounds += 1

            for tc in raw_tool_calls:
                fn = tc["function"]
                name: str = fn["name"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}

                if self._on_tool_call is not None:
                    should_continue = self._on_tool_call(name, args)
                    if not should_continue:
                        return ConversationResponse(
                            content="",
                            tool_calls=all_tool_calls,
                            usage=final_usage,
                        )

                result = await self._tools.call(name, args)

                if self._on_tool_result is not None:
                    self._on_tool_result(name, result)

                all_tool_calls.append({"name": name, "args": args, "result": result})
                self._history.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_calls": None,
                    "tool_call_id": tc.get("id"),
                })

    def _build_messages(self) -> list[ChatMessage]:
        if self._system:
            return [
                {"role": "system", "content": self._system, "tool_calls": None, "tool_call_id": None},
                *self._history,
            ]
        return list(self._history)
