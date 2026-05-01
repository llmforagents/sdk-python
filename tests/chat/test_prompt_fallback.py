"""Unit tests for the prompt-mode fallback parser/formatter."""
from __future__ import annotations
import json
from llm4agents.chat.prompt_fallback import (
    format_tools_for_prompt,
    parse_tool_calls_from_text,
)


TOOLS_MCP = [
    {
        "name": "google_search",
        "description": "Search Google",
        "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
    {
        "name": "fetch_url",
        "description": "Fetch a URL",
        "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}},
    },
]


def test_format_tools_for_prompt_includes_header_and_each_tool():
    out = format_tools_for_prompt(TOOLS_MCP)
    assert "<tool_call>" in out
    assert "google_search: Search Google" in out
    assert "fetch_url: Fetch a URL" in out
    assert "properties" in out


def test_format_tools_for_prompt_accepts_openai_shape():
    """Also accept the OpenAI ``{type:function, function:{...}}`` shape."""
    openai_shape = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Adds numbers",
                "parameters": {"type": "object"},
            },
        }
    ]
    out = format_tools_for_prompt(openai_shape)
    assert "calc: Adds numbers" in out


def test_parse_single_tool_call():
    text = (
        'Sure thing.\n<tool_call>\n{"name":"google_search","arguments":{"q":"bitcoin"}}\n'
        "</tool_call>"
    )
    calls, remainder = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "google_search"
    # arguments are always JSON-stringified
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "bitcoin"}
    assert calls[0]["id"].startswith("pmpt_")
    assert calls[0]["type"] == "function"
    assert remainder == "Sure thing."


def test_parse_multiple_tool_calls_and_skip_malformed():
    text = (
        '<tool_call>not-json</tool_call>\n'
        '<tool_call>{"name":"a","arguments":{}}</tool_call>\n'
        '<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>'
    )
    calls, remainder = parse_tool_calls_from_text(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"x": 1}
    assert remainder == ""


def test_parse_returns_empty_when_no_blocks():
    calls, remainder = parse_tool_calls_from_text("Just a regular answer.")
    assert calls == []
    assert remainder == "Just a regular answer."
