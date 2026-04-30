from __future__ import annotations
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Callable
from llm4agents.transport.http import HttpTransport
from llm4agents.chat.types import ChatMessage, ResponseMeta
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
        self._on_round_meta: Callable[[ResponseMeta], None] | None = opts.get("on_round_meta")
        self._on_tools_ignored: Callable[[str], None] | None = opts.get("on_tools_ignored")
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
        if self._on_round_meta is not None:
            opts["on_round_meta"] = self._on_round_meta
        if self._on_tools_ignored is not None:
            opts["on_tools_ignored"] = self._on_tools_ignored
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
        round_count = 0

        while True:
            params: dict[str, Any] = {
                "model": self._model,
                "messages": self._build_messages(),
            }
            tools_sent = False
            if self._tools is not None:
                if not self._tools.definitions:
                    await self._tools.fetch_definitions()
                params["tools"] = self._tools.definitions
                tools_sent = True

            data, headers = await self._http.post_with_meta("/v1/chat/completions", params)

            if self._on_round_meta is not None:
                self._on_round_meta(ResponseMeta.from_headers(headers))

            choice = data["choices"][0]
            msg: ChatMessage = choice["message"]
            usage: dict[str, int] = data.get("usage") or {}
            final_usage = usage
            finish_reason: str = choice.get("finish_reason", "stop")

            self._history.append(msg)

            raw_tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []

            if not raw_tool_calls or finish_reason == "stop":
                if round_count == 0 and tools_sent and not raw_tool_calls and self._on_tools_ignored is not None:
                    self._on_tools_ignored(self._model)
                return ConversationResponse(
                    content=msg.get("content") or "",
                    tool_calls=all_tool_calls,
                    usage=final_usage,
                )

            self._check_tool_limit()
            self._tool_rounds += 1
            round_count += 1

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
                    "content": result.text,
                    "tool_calls": None,
                    "tool_call_id": tc.get("id"),
                })

    async def stream(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Async generator that yields stream events.

        Event shapes:
          {"type": "text", "content": str}
          {"type": "reasoning", "content": str}
          {"type": "tool_call", "name": str, "args": dict}
          {"type": "tool_result", "name": str, "result": McpToolResult}
          {"type": "meta", "meta": ResponseMeta}
          {"type": "done", "response": {"content": str, "tool_calls": list, "usage": dict}}
        """
        self._history.append(
            {"role": "user", "content": message, "tool_calls": None, "tool_call_id": None}
        )

        all_tool_calls: list[dict[str, Any]] = []
        round_count = 0
        full_content = ""

        while True:
            params: dict[str, Any] = {
                "model": self._model,
                "messages": self._build_messages(),
                "stream": True,
            }
            tools_sent = False
            if self._tools is not None:
                if not self._tools.definitions:
                    await self._tools.fetch_definitions()
                params["tools"] = self._tools.definitions
                tools_sent = True

            headers, sse_stream = await self._http.post_stream_with_meta(
                "/v1/chat/completions", params
            )

            streamed_content = ""
            pending_tool_calls: dict[int, dict[str, str]] = {}
            chunk_usage: dict[str, int] | None = None

            async for chunk in sse_stream:
                choices = chunk.get("choices") or []
                first_choice = choices[0] if choices else None
                delta = first_choice.get("delta") if first_choice else None
                if delta is None:
                    if chunk.get("usage"):
                        chunk_usage = chunk["usage"]
                    continue

                if delta.get("reasoning"):
                    yield {"type": "reasoning", "content": delta["reasoning"]}

                if delta.get("content"):
                    streamed_content += delta["content"]
                    full_content += delta["content"]
                    yield {"type": "text", "content": delta["content"]}

                for tc_delta in delta.get("tool_calls") or []:
                    idx: int = tc_delta.get("index", 0)
                    if idx not in pending_tool_calls:
                        pending_tool_calls[idx] = {
                            "id": tc_delta.get("id") or "",
                            "name": (tc_delta.get("function") or {}).get("name") or "",
                            "args": (tc_delta.get("function") or {}).get("arguments") or "",
                        }
                    else:
                        existing = pending_tool_calls[idx]
                        if tc_delta.get("id"):
                            existing["id"] = tc_delta["id"]
                        fn_part = tc_delta.get("function") or {}
                        if fn_part.get("name"):
                            existing["name"] = fn_part["name"]
                        if fn_part.get("arguments") is not None:
                            existing["args"] += fn_part["arguments"]

                if chunk.get("usage"):
                    chunk_usage = chunk["usage"]

            round_meta = ResponseMeta.from_headers(headers)
            yield {"type": "meta", "meta": round_meta}
            if self._on_round_meta is not None:
                self._on_round_meta(round_meta)

            tool_calls_list = list(pending_tool_calls.values())

            # Build assistant message
            assistant_msg: ChatMessage = {
                "role": "assistant",
                "content": streamed_content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["args"]},
                    }
                    for tc in tool_calls_list
                ] or None,
                "tool_call_id": None,
            }
            self._history.append(assistant_msg)

            if not tool_calls_list:
                if round_count == 0 and tools_sent and self._on_tools_ignored is not None:
                    self._on_tools_ignored(self._model)
                usage_dict: dict[str, int] = {}
                if chunk_usage:
                    usage_dict = {
                        "prompt_tokens": chunk_usage.get("prompt_tokens", 0),
                        "completion_tokens": chunk_usage.get("completion_tokens", 0),
                    }
                yield {
                    "type": "done",
                    "response": {
                        "content": full_content,
                        "tool_calls": all_tool_calls,
                        "usage": usage_dict,
                    },
                }
                return

            round_count += 1
            self._check_tool_limit()
            self._tool_rounds += 1

            for tc in tool_calls_list:
                name: str = tc["name"]
                try:
                    args = json.loads(tc.get("args") or "{}")
                except json.JSONDecodeError:
                    args = {}

                yield {"type": "tool_call", "name": name, "args": args}

                if self._on_tool_call is not None:
                    should_continue = self._on_tool_call(name, args)
                    if not should_continue:
                        yield {
                            "type": "done",
                            "response": {
                                "content": full_content,
                                "tool_calls": all_tool_calls,
                                "usage": {},
                            },
                        }
                        return

                result = await self._tools.call(name, args)

                if self._on_tool_result is not None:
                    self._on_tool_result(name, result)

                all_tool_calls.append({"name": name, "args": args, "result": result})
                self._history.append({
                    "role": "tool",
                    "content": result.text,
                    "tool_calls": None,
                    "tool_call_id": tc.get("id"),
                })
                yield {"type": "tool_result", "name": name, "result": result}

            # Image short-circuit
            has_image = any(
                c.type == "image"
                for tc_rec in all_tool_calls
                for c in tc_rec["result"].content
            )
            if has_image:
                yield {
                    "type": "done",
                    "response": {
                        "content": full_content,
                        "tool_calls": all_tool_calls,
                        "usage": {},
                    },
                }
                return

            full_content = ""

    def _build_messages(self) -> list[ChatMessage]:
        if self._system:
            return [
                {"role": "system", "content": self._system, "tool_calls": None, "tool_call_id": None},
                *self._history,
            ]
        return list(self._history)
