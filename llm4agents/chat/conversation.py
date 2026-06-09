from __future__ import annotations
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Callable
from llm4agents.transport.http import HttpTransport
from llm4agents.chat.types import ChatMessage, ResponseMeta
from llm4agents.chat.prompt_fallback import (
    format_tools_for_prompt,
    parse_tool_calls_from_text,
)
from llm4agents.errors import LLM4AgentsError


def _normalize_tool_calls(
    tool_calls: list[dict[str, Any]], round_count: int
) -> list[dict[str, Any]]:
    """Synthesize id when provider omits it. Some providers (Gemini, certain
    Anthropic models via OpenRouter) return tool_calls without an id, which
    breaks the next round because role:'tool' messages need tool_call_id."""
    out: list[dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        existing_id = tc.get("id")
        if not existing_id:
            new_tc = dict(tc)
            new_tc["id"] = f"auto_{round_count}_{i}_{int(time.time() * 1000)}"
            out.append(new_tc)
        else:
            out.append(tc)
    return out


@dataclass(frozen=True)
class ConversationResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)


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
        self._enable_prompt_tool_fallback: bool = bool(
            opts.get("enable_prompt_tool_fallback", False)
        )
        self._max_tool_rounds: int = opts.get("max_tool_rounds", 10)
        # First-round-only tool selection. Mirrors the TS SDK semantics: the
        # caller's `tool_choice` is forwarded to the LLM on round 1, then the
        # field is dropped on every subsequent round so the model can wrap up
        # with plain text once its forced tool has returned. Without this
        # auto-revert, `'required'` on every round forces the model to keep
        # tool-calling forever and the conversation hits `max_tool_rounds`;
        # Anthropic also 400s when `tool_choice='required'` is paired with a
        # turn that has no remaining tool work. See README for the rationale.
        self._tool_choice: Any = opts.get("tool_choice")
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
            "enable_prompt_tool_fallback": self._enable_prompt_tool_fallback,
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
        if self._tool_choice is not None:
            opts["tool_choice"] = self._tool_choice
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
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        round_count = 0

        while True:
            tool_defs: list[dict[str, Any]] | None = None
            if self._tools is not None:
                if not self._tools.definitions:
                    await self._tools.fetch_definitions()
                tool_defs = list(self._tools.definitions)

            params: dict[str, Any] = {
                "model": self._model,
                "messages": self._build_messages(),
            }
            if tool_defs:
                params["tools"] = tool_defs
            # First-round-only tool_choice — see __init__ for the rationale.
            if round_count == 0 and self._tool_choice is not None:
                params["tool_choice"] = self._tool_choice

            data, headers = await self._http.post_with_meta("/v1/chat/completions", params)

            if self._on_round_meta is not None:
                self._on_round_meta(ResponseMeta.from_headers(headers))

            usage: dict[str, Any] = data.get("usage") or {}
            total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            r_tok = usage.get("reasoning_tokens")
            if r_tok is not None:
                total_reasoning_tokens += int(r_tok)

            choice = data["choices"][0]
            msg: ChatMessage = choice["message"]
            # BUG-08: normalize content: null → "" so the next request's
            # messages[*].content stays a string for strict backends.
            if msg.get("content") is None:
                msg = {**msg, "content": ""}
            # BUG-09: synthesize tool_call.id when provider omits it
            # (Gemini, certain Anthropic models via OpenRouter). Without an id
            # the matching role:'tool' reply lacks tool_call_id and breaks
            # the next round.
            existing_tcs = msg.get("tool_calls")
            if existing_tcs:
                msg = {**msg, "tool_calls": _normalize_tool_calls(existing_tcs, round_count)}
            self._history.append(msg)

            raw_tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []

            if not raw_tool_calls:
                ignored_tools = round_count == 0 and bool(tool_defs)
                if ignored_tools and self._on_tools_ignored is not None:
                    self._on_tools_ignored(self._model)

                # Prompt-mode fallback: retry round 0 with tools described
                # in the system prompt, then parse <tool_call> blocks.
                if ignored_tools and self._enable_prompt_tool_fallback and tool_defs:
                    fb = await self._run_prompt_fallback_round(tool_defs)
                    total_prompt_tokens += fb["usage"]["prompt_tokens"]
                    total_completion_tokens += fb["usage"]["completion_tokens"]
                    if fb["usage"].get("reasoning_tokens") is not None:
                        total_reasoning_tokens += fb["usage"]["reasoning_tokens"]

                    if fb["tool_calls"]:
                        # Replace the ignored assistant message with the prompt-mode one
                        self._history.pop()
                        self._history.append(fb["assistant_message"])
                        for tc in fb["tool_calls"]:
                            await self._execute_tool_call(tc, all_tool_calls)
                        round_count += 1
                        self._check_tool_limit()
                        self._tool_rounds += 1
                        continue

                    # Fallback also produced no tool calls → return prompt-mode text
                    self._history.pop()
                    self._history.append(fb["assistant_message"])
                    return ConversationResponse(
                        content=fb["text_without_blocks"],
                        tool_calls=all_tool_calls,
                        usage=self._build_usage_dict(
                            total_prompt_tokens, total_completion_tokens, total_reasoning_tokens
                        ),
                    )

                content_val = msg.get("content")
                content_str = content_val if isinstance(content_val, str) else ""
                return ConversationResponse(
                    content=content_str,
                    tool_calls=all_tool_calls,
                    usage=self._build_usage_dict(
                        total_prompt_tokens, total_completion_tokens, total_reasoning_tokens
                    ),
                )

            self._check_tool_limit()
            self._tool_rounds += 1
            round_count += 1

            for tc in raw_tool_calls:
                await self._execute_tool_call(tc, all_tool_calls)

    async def _execute_tool_call(
        self, tc: dict[str, Any], all_tool_calls: list[dict[str, Any]]
    ) -> None:
        fn = tc["function"]
        name: str = fn["name"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}

        if self._on_tool_call is not None:
            should_continue = self._on_tool_call(name, args)
            if not should_continue:
                return

        result = await self._tools.call(name, args)

        if self._on_tool_result is not None:
            self._on_tool_result(name, result)

        all_tool_calls.append({"name": name, "args": args, "result": result})
        self._history.append(
            {
                "role": "tool",
                "content": result.text,
                "tool_calls": None,
                "tool_call_id": tc.get("id"),
                "name": name,
            }
        )

    async def stream(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Async generator that yields stream events.

        Event shapes:
          {"type": "text", "content": str}
          {"type": "reasoning", "content": str}
          {"type": "tool_call", "name": str, "args": dict}
          {"type": "tool_result", "name": str, "result": McpToolResult}
          {"type": "meta", "meta": ResponseMeta}
          {"type": "fallback", "reason": "tools_ignored", "model": str}
          {"type": "x402_receipt", "transaction": str, "network": str, "amount": str, "payer": str}
          {"type": "done", "response": {"content": str, "tool_calls": list, "usage": dict}}

        ``x402_receipt`` only fires when the client is constructed in
        x402 walk-up mode (``payment=PaymentConfig(mode='x402', ...)``).
        The proxy emits a trailing ``event: x402-receipt`` SSE chunk
        after settlement; we always yield it BEFORE the matching ``done``
        event so consumers can correlate the receipt with the round.
        """
        self._history.append(
            {"role": "user", "content": message, "tool_calls": None, "tool_call_id": None}
        )

        all_tool_calls: list[dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        round_count = 0
        full_content = ""

        while True:
            tool_defs: list[dict[str, Any]] | None = None
            if self._tools is not None:
                if not self._tools.definitions:
                    await self._tools.fetch_definitions()
                tool_defs = list(self._tools.definitions)

            params: dict[str, Any] = {
                "model": self._model,
                "messages": self._build_messages(),
                "stream": True,
            }
            if tool_defs:
                params["tools"] = tool_defs
            # First-round-only tool_choice — see __init__ for the rationale.
            if round_count == 0 and self._tool_choice is not None:
                params["tool_choice"] = self._tool_choice

            headers, sse_stream = await self._http.post_stream_with_meta(
                "/v1/chat/completions", params
            )

            streamed_content = ""
            pending_tool_calls: dict[int, dict[str, str]] = {}
            chunk_usage: dict[str, int] | None = None
            # Captured from the transport when the proxy emits a trailing
            # `event: x402-receipt` SSE chunk. Yielded as
            # `{"type": "x402_receipt", ...}` BEFORE each `done` site so
            # consumers can correlate the receipt with the chat round.
            received_receipt: dict[str, str] | None = None

            async for chunk in sse_stream:
                event_name = chunk.get("_event") if isinstance(chunk, dict) else None
                if event_name == "x402-receipt":
                    data = chunk.get("data") or {}
                    if all(
                        isinstance(data.get(k), str)
                        for k in ("transaction", "network", "amount", "payer")
                    ):
                        received_receipt = {
                            "transaction": data["transaction"],
                            "network": data["network"],
                            "amount": data["amount"],
                            "payer": data["payer"],
                        }
                    continue
                if event_name is not None:
                    # Unknown typed SSE event — forward-compatible skip.
                    continue
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

            if chunk_usage:
                total_prompt_tokens += int(chunk_usage.get("prompt_tokens", 0) or 0)
                total_completion_tokens += int(chunk_usage.get("completion_tokens", 0) or 0)
                r_tok = chunk_usage.get("reasoning_tokens")
                if r_tok is not None:
                    total_reasoning_tokens += int(r_tok)

            round_meta = ResponseMeta.from_headers(headers)
            # Streaming responses don't carry x-cost-usd-cents in the
            # response headers — by the time headers flush, the proxy
            # hasn't finished tallying the round. The terminating SSE
            # chunk's `usage.cost` field (USD float) carries the final
            # number instead; promote it into cost_usd_cents (× 100)
            # so streaming consumers see the same field as non-streaming.
            # Header-based path is preserved: if cost_usd_cents was
            # already populated from headers, leave it.
            chunk_cost = chunk_usage.get("cost") if chunk_usage else None
            if round_meta.cost_usd_cents is None and isinstance(chunk_cost, (int, float)):
                from dataclasses import replace as _dc_replace
                round_meta = _dc_replace(round_meta, cost_usd_cents=chunk_cost * 100)
            yield {"type": "meta", "meta": round_meta}
            if self._on_round_meta is not None:
                self._on_round_meta(round_meta)

            tool_calls_list = list(pending_tool_calls.values())

            # BUG-09: normalize streamed tool_calls — providers like Gemini
            # may emit chunks without id, leading to empty tool_call_id on
            # the matching role:'tool' message and a broken next round.
            normalized_tool_calls = _normalize_tool_calls(
                [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["args"]},
                    }
                    for tc in tool_calls_list
                ],
                round_count,
            )
            # Sync synthesized ids back into tool_calls_list so the matching
            # role:'tool' history entries get the same tool_call_id we put on
            # the assistant message.
            for tc, norm in zip(tool_calls_list, normalized_tool_calls):
                tc["id"] = norm["id"]

            # BUG-08: assistant message content stays a plain string.
            # Always coerce to str so the next request can include it as a
            # valid messages[*].content entry.
            assistant_msg: ChatMessage = {
                "role": "assistant",
                "content": streamed_content,
                "tool_calls": normalized_tool_calls or None,
                "tool_call_id": None,
            }
            self._history.append(assistant_msg)

            if not tool_calls_list:
                ignored_tools = round_count == 0 and bool(tool_defs)
                if ignored_tools and self._on_tools_ignored is not None:
                    self._on_tools_ignored(self._model)

                # Prompt-mode fallback (non-streamed): rerun round 0 with tools
                # injected into the system prompt, then execute parsed calls.
                if ignored_tools and self._enable_prompt_tool_fallback and tool_defs:
                    yield {"type": "fallback", "reason": "tools_ignored", "model": self._model}

                    self._history.pop()  # drop the failed first attempt
                    fb = await self._run_prompt_fallback_round(tool_defs)
                    total_prompt_tokens += fb["usage"]["prompt_tokens"]
                    total_completion_tokens += fb["usage"]["completion_tokens"]
                    if fb["usage"].get("reasoning_tokens") is not None:
                        total_reasoning_tokens += fb["usage"]["reasoning_tokens"]
                    self._history.append(fb["assistant_message"])

                    if not fb["tool_calls"]:
                        yield {"type": "text", "content": fb["text_without_blocks"]}
                        if received_receipt is not None:
                            yield {"type": "x402_receipt", **received_receipt}
                        yield {
                            "type": "done",
                            "response": {
                                "content": fb["text_without_blocks"],
                                "tool_calls": all_tool_calls,
                                "usage": self._build_usage_dict(
                                    total_prompt_tokens,
                                    total_completion_tokens,
                                    total_reasoning_tokens,
                                ),
                            },
                        }
                        return

                    for tc in fb["tool_calls"]:
                        fn = tc["function"]
                        name = fn["name"]
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        yield {"type": "tool_call", "name": name, "args": args}

                        if self._on_tool_call is not None:
                            should_continue = self._on_tool_call(name, args)
                            if not should_continue:
                                if received_receipt is not None:
                                    yield {"type": "x402_receipt", **received_receipt}
                                yield {
                                    "type": "done",
                                    "response": {
                                        "content": full_content,
                                        "tool_calls": all_tool_calls,
                                        "usage": self._build_usage_dict(
                                            total_prompt_tokens,
                                            total_completion_tokens,
                                            total_reasoning_tokens,
                                        ),
                                    },
                                }
                                return

                        result = await self._tools.call(name, args)
                        if self._on_tool_result is not None:
                            self._on_tool_result(name, result)
                        all_tool_calls.append({"name": name, "args": args, "result": result})
                        self._history.append(
                            {
                                "role": "tool",
                                "content": result.text,
                                "tool_calls": None,
                                "tool_call_id": tc.get("id"),
                                "name": name,
                            }
                        )
                        yield {"type": "tool_result", "name": name, "result": result}

                    round_count += 1
                    self._check_tool_limit()
                    self._tool_rounds += 1
                    full_content = ""
                    continue

                if received_receipt is not None:
                    yield {"type": "x402_receipt", **received_receipt}
                yield {
                    "type": "done",
                    "response": {
                        "content": full_content,
                        "tool_calls": all_tool_calls,
                        "usage": self._build_usage_dict(
                            total_prompt_tokens, total_completion_tokens, total_reasoning_tokens
                        ),
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
                        if received_receipt is not None:
                            yield {"type": "x402_receipt", **received_receipt}
                        yield {
                            "type": "done",
                            "response": {
                                "content": full_content,
                                "tool_calls": all_tool_calls,
                                "usage": self._build_usage_dict(
                                    total_prompt_tokens,
                                    total_completion_tokens,
                                    total_reasoning_tokens,
                                ),
                            },
                        }
                        return

                result = await self._tools.call(name, args)

                if self._on_tool_result is not None:
                    self._on_tool_result(name, result)

                all_tool_calls.append({"name": name, "args": args, "result": result})
                self._history.append(
                    {
                        "role": "tool",
                        "content": result.text,
                        "tool_calls": None,
                        "tool_call_id": tc.get("id"),
                        "name": name,
                    }
                )
                yield {"type": "tool_result", "name": name, "result": result}

            # Image short-circuit
            has_image = any(
                c.type == "image"
                for tc_rec in all_tool_calls
                for c in tc_rec["result"].content
            )
            if has_image:
                if received_receipt is not None:
                    yield {"type": "x402_receipt", **received_receipt}
                yield {
                    "type": "done",
                    "response": {
                        "content": full_content,
                        "tool_calls": all_tool_calls,
                        "usage": self._build_usage_dict(
                            total_prompt_tokens, total_completion_tokens, total_reasoning_tokens
                        ),
                    },
                }
                return

            full_content = ""

    async def _run_prompt_fallback_round(
        self, tool_defs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Run a single non-streamed round with tools described in the system prompt.

        Parses ``<tool_call>...</tool_call>`` blocks from the model's reply and
        returns the assistant message (always with ``content`` as a string),
        the parsed tool calls, the text minus the blocks, and the round usage.
        """
        prompt_tools_block = format_tools_for_prompt(tool_defs)
        augmented_system = (
            f"{self._system}\n\n{prompt_tools_block}"
            if self._system
            else prompt_tools_block
        )
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": augmented_system,
                "tool_calls": None,
                "tool_call_id": None,
            },
            *self._history,
        ]
        params: dict[str, Any] = {"model": self._model, "messages": messages}

        data, headers = await self._http.post_with_meta("/v1/chat/completions", params)

        if self._on_round_meta is not None:
            self._on_round_meta(ResponseMeta.from_headers(headers))

        choice = data["choices"][0]
        raw_content = choice["message"].get("content")
        text = raw_content if isinstance(raw_content, str) else ""
        tool_calls, text_without_blocks = parse_tool_calls_from_text(text)

        # BUG-08: content stays a string even when tool_calls are present.
        assistant_message: ChatMessage = {
            "role": "assistant",
            "content": text_without_blocks,
            "tool_calls": tool_calls or None,
            "tool_call_id": None,
        }

        usage: dict[str, Any] = data.get("usage") or {}
        usage_out: dict[str, int | None] = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        }
        r_tok = usage.get("reasoning_tokens")
        usage_out["reasoning_tokens"] = int(r_tok) if r_tok is not None else None

        return {
            "assistant_message": assistant_message,
            "tool_calls": tool_calls,
            "text_without_blocks": text_without_blocks,
            "usage": usage_out,
        }

    @staticmethod
    def _build_usage_dict(prompt: int, completion: int, reasoning: int) -> dict[str, int]:
        out: dict[str, int] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
        if reasoning > 0:
            out["reasoning_tokens"] = reasoning
        return out

    def _build_messages(self) -> list[ChatMessage]:
        if self._system:
            return [
                {
                    "role": "system",
                    "content": self._system,
                    "tool_calls": None,
                    "tool_call_id": None,
                },
                *self._history,
            ]
        return list(self._history)
