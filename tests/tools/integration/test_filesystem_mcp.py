"""Live integration: spawn @modelcontextprotocol/server-filesystem and read a file.

Opt-in: set MCP_LIVE=1. Requires `npx` in PATH.
"""
import os
import pytest
import tempfile
from pathlib import Path

from llm4agents.client import LLM4AgentsClient
from llm4agents.transport.mcp_stdio import StdioServerConfig

live = os.environ.get("MCP_LIVE") == "1"
pytestmark = pytest.mark.skipif(not live, reason="opt-in: set MCP_LIVE=1")


@pytest.mark.asyncio
async def test_filesystem_server_lists_and_reads(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("hello from MCP live (py)")

    client = LLM4AgentsClient(api_key=os.environ.get("LLM4AGENTS_API_KEY", "sk-proxy-" + "a" * 40))
    try:
        await client.tools.connect(StdioServerConfig(
            name="fs",
            command="npx",
            args=("-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)),
        ))
        # Tool name varies by server-filesystem version: 'read_file' vs 'read_text_file'
        tools = await client.tools._servers["fs"].list_tools()
        tool_names = [t["name"] for t in tools]
        assert any(name in ("read_file", "read_text_file") for name in tool_names)

        read_tool = "read_text_file" if "read_text_file" in tool_names else "read_file"
        result = await client.tools.call_tool(read_tool, {"path": "hello.txt"}, server="fs")
        # McpToolResult shape — verify it contains the expected text
        text_parts = [c.text for c in result.content if hasattr(c, "text")]
        joined = " ".join(text_parts)
        assert "hello from MCP live" in joined
    finally:
        await client.close()
