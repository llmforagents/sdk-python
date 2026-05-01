"""Prompt-mode fallback for models without native function-calling.

When ``Conversation`` is configured with ``enable_prompt_tool_fallback=True``
and round 0 returns no ``tool_calls`` despite tools being sent, the SDK
retries the round with the tool definitions injected into the system prompt.
The model can then emit ``<tool_call>{"name":"...","arguments":{...}}</tool_call>``
blocks which are parsed back into the standard tool-call shape so the loop
can continue uniformly with native tool callers.
"""
from __future__ import annotations
import json
import re
from typing import Any

_FALLBACK_INSTRUCTIONS = """You have access to the following tools. To call a tool, output a fenced block exactly like this:

<tool_call>
{"name": "tool_name", "arguments": {"key": "value"}}
</tool_call>

You may emit zero or more <tool_call> blocks. After all tool calls, the system will execute them and reply with the results so you can produce a final answer. If no tool is needed, answer the user directly without any <tool_call> block.

Available tools:"""


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")


def format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    """Render an MCP-style tool list as a system-prompt-friendly block.

    Accepts either the MCP tool shape (``{"name", "description", "inputSchema"}``)
    used by ``llm4agents.tools.types.ToolDefinition`` or the OpenAI shape
    (``{"type": "function", "function": {"name", "description", "parameters"}}``).
    """
    lines = [_FALLBACK_INSTRUCTIONS]
    for t in tools:
        if "function" in t and isinstance(t.get("function"), dict):
            fn = t["function"]
            name = fn.get("name", "")
            description = fn.get("description", "")
            params = fn.get("parameters", {})
        else:
            name = t.get("name", "")
            description = t.get("description", "")
            params = t.get("inputSchema", {})
        lines.append(f"\n- {name}: {description}")
        lines.append(f"  parameters: {json.dumps(params)}")
    return "\n".join(lines)


def parse_tool_calls_from_text(
    text: str, id_prefix: str = "pmpt"
) -> tuple[list[dict[str, Any]], str]:
    """Extract ``<tool_call>...</tool_call>`` blocks from a model's text reply.

    Returns ``(tool_calls, text_without_blocks)`` where each tool call is shaped
    like a native OpenAI ``tool_calls`` entry (``arguments`` is a JSON string).
    Malformed JSON inside a block is skipped silently — the loop falls through
    to "no tool calls parsed" which the caller treats as a regular text answer.
    """
    calls: list[dict[str, Any]] = []
    idx = 0
    for match in _TOOL_CALL_BLOCK.finditer(text):
        body = match.group(1) or ""
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name")
        if not isinstance(name, str):
            continue
        raw_args = parsed.get("arguments")
        if isinstance(raw_args, str):
            args_string = raw_args
        else:
            args_string = json.dumps(raw_args if raw_args is not None else {})
        calls.append(
            {
                "id": f"{id_prefix}_{idx}",
                "type": "function",
                "function": {"name": name, "arguments": args_string},
            }
        )
        idx += 1
    text_without_blocks = _TOOL_CALL_BLOCK.sub("", text).strip()
    return calls, text_without_blocks
