"""Unit tests for connect_stdio with mocked mcp.client modules."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm4agents.transport.mcp_stdio import StdioServerConfig, connect_stdio


@pytest.mark.asyncio
async def test_connect_stdio_lists_tools() -> None:
    fake_tool = MagicMock(description="echo tool", inputSchema={"type": "object"})
    fake_tool.name = "echo"  # MagicMock auto-creates .name as a Mock; set explicitly

    fake_session = MagicMock()
    fake_session.initialize = AsyncMock(return_value=None)
    fake_session.list_tools = AsyncMock(return_value=MagicMock(tools=[fake_tool]))

    fake_stdio_ctx = MagicMock()
    fake_stdio_ctx.__aenter__ = AsyncMock(return_value=("read", "write"))
    fake_stdio_ctx.__aexit__ = AsyncMock(return_value=None)

    fake_session_ctx = MagicMock()
    fake_session_ctx.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session_ctx.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("llm4agents.transport.mcp_stdio.stdio_client", return_value=fake_stdio_ctx),
        patch("llm4agents.transport.mcp_stdio.ClientSession", return_value=fake_session_ctx),
    ):
        handle = await connect_stdio(StdioServerConfig(name="fs", command="echo"))
        tools = await handle.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"
        await handle.disconnect()


@pytest.mark.asyncio
async def test_connect_stdio_call_tool_returns_content() -> None:
    fake_content_item = MagicMock(type="text", text="hello")

    fake_session = MagicMock()
    fake_session.initialize = AsyncMock(return_value=None)
    fake_session.call_tool = AsyncMock(
        return_value=MagicMock(content=[fake_content_item])
    )

    fake_stdio_ctx = MagicMock()
    fake_stdio_ctx.__aenter__ = AsyncMock(return_value=("r", "w"))
    fake_stdio_ctx.__aexit__ = AsyncMock(return_value=None)

    fake_session_ctx = MagicMock()
    fake_session_ctx.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session_ctx.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("llm4agents.transport.mcp_stdio.stdio_client", return_value=fake_stdio_ctx),
        patch("llm4agents.transport.mcp_stdio.ClientSession", return_value=fake_session_ctx),
    ):
        handle = await connect_stdio(StdioServerConfig(name="fs", command="echo"))
        result = await handle.call_tool("echo", {"x": 1})
        assert result["content"][0]["text"] == "hello"
        await handle.disconnect()


@pytest.mark.asyncio
async def test_connect_stdio_timeout() -> None:
    """If the underlying connect hangs, asyncio.wait_for raises asyncio.TimeoutError."""

    async def _hang(*_args: object, **_kwargs: object) -> tuple[str, str]:
        await asyncio.sleep(10)
        return ("r", "w")  # unreachable

    fake_stdio_ctx = MagicMock()
    fake_stdio_ctx.__aenter__ = AsyncMock(side_effect=_hang)
    fake_stdio_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("llm4agents.transport.mcp_stdio.stdio_client", return_value=fake_stdio_ctx):
        with pytest.raises(asyncio.TimeoutError):
            await connect_stdio(
                StdioServerConfig(name="fs", command="echo"),
                connect_timeout_s=0.05,
            )


@pytest.mark.asyncio
async def test_connect_stdio_cleanup_on_init_failure() -> None:
    """Resources are cleaned up when session.initialize() raises."""
    fake_session = MagicMock()
    fake_session.initialize = AsyncMock(side_effect=RuntimeError("init failed"))

    fake_stdio_ctx = MagicMock()
    fake_stdio_ctx.__aenter__ = AsyncMock(return_value=("r", "w"))
    fake_stdio_ctx.__aexit__ = AsyncMock(return_value=None)

    fake_session_ctx = MagicMock()
    fake_session_ctx.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session_ctx.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("llm4agents.transport.mcp_stdio.stdio_client", return_value=fake_stdio_ctx),
        patch("llm4agents.transport.mcp_stdio.ClientSession", return_value=fake_session_ctx),
    ):
        with pytest.raises(RuntimeError, match="init failed"):
            await connect_stdio(StdioServerConfig(name="fs", command="echo"))

    # __aexit__ called proves the exit stack cleaned up the stdio context
    fake_stdio_ctx.__aexit__.assert_called_once()
