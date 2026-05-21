"""Unit tests for the Tools registry + dispatcher (mocked transports)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm4agents.tools.tools import Tools
from llm4agents.transport.mcp_stdio import StdioServerConfig
from llm4agents.transport.mcp_http import HttpServerConfig


def _make_handle(name: str, content_text: str) -> MagicMock:
    handle = MagicMock()
    handle.name = name
    handle.list_tools = AsyncMock(return_value=[])
    handle.call_tool = AsyncMock(
        return_value={"content": [{"type": "text", "text": content_text}]}
    )
    handle.disconnect = AsyncMock(return_value=None)
    return handle


@pytest.fixture
def fake_mcp() -> MagicMock:
    mcp = MagicMock()
    mcp.call_tool = AsyncMock(
        return_value=MagicMock(content=(), text="proxy-result")
    )
    return mcp


async def test_connect_stores_in_registry(fake_mcp: MagicMock) -> None:
    tools = Tools(fake_mcp)
    fake_handle = _make_handle("fs", "stdio-result")
    with patch("llm4agents.tools.tools._connect", AsyncMock(return_value=fake_handle)):
        handle = await tools.connect(StdioServerConfig(name="fs", command="echo"))
    assert handle is fake_handle
    assert "fs" in tools.connected_servers()


async def test_connect_duplicate_name_raises(fake_mcp: MagicMock) -> None:
    tools = Tools(fake_mcp)
    with patch(
        "llm4agents.tools.tools._connect",
        AsyncMock(return_value=_make_handle("fs", "x")),
    ):
        await tools.connect(StdioServerConfig(name="fs", command="echo"))
        with pytest.raises(ValueError, match="already connected"):
            await tools.connect(StdioServerConfig(name="fs", command="echo"))


async def test_disconnect_removes_from_registry(fake_mcp: MagicMock) -> None:
    tools = Tools(fake_mcp)
    handle = _make_handle("fs", "x")
    with patch("llm4agents.tools.tools._connect", AsyncMock(return_value=handle)):
        await tools.connect(StdioServerConfig(name="fs", command="echo"))
    await tools.disconnect("fs")
    assert tools.connected_servers() == []
    handle.disconnect.assert_called_once()


async def test_disconnect_all_clears(fake_mcp: MagicMock) -> None:
    tools = Tools(fake_mcp)
    h1 = _make_handle("a", "x")
    h2 = _make_handle("b", "y")
    handles_iter = iter([h1, h2])

    async def _side_effect(cfg: object) -> MagicMock:
        return next(handles_iter)

    with patch("llm4agents.tools.tools._connect", side_effect=_side_effect):
        await tools.connect(StdioServerConfig(name="a", command="echo"))
        await tools.connect(HttpServerConfig(name="b", url="https://x"))
    await tools.disconnect_all()
    assert tools.connected_servers() == []
    h1.disconnect.assert_called_once()
    h2.disconnect.assert_called_once()


async def test_call_tool_dispatches_to_named_server(fake_mcp: MagicMock) -> None:
    tools = Tools(fake_mcp)
    handle = _make_handle("fs", "stdio-result")
    with patch("llm4agents.tools.tools._connect", AsyncMock(return_value=handle)):
        await tools.connect(StdioServerConfig(name="fs", command="echo"))
    result = await tools.call_tool("read", {"path": "x"}, server="fs")
    handle.call_tool.assert_called_once_with("read", {"path": "x"})
    assert result.text == "stdio-result"


async def test_call_tool_unknown_server_raises(fake_mcp: MagicMock) -> None:
    tools = Tools(fake_mcp)
    with pytest.raises(ValueError, match="not connected"):
        await tools.call_tool("any", {}, server="nonexistent")


async def test_call_tool_without_server_uses_proxy_gateway(fake_mcp: MagicMock) -> None:
    tools = Tools(fake_mcp)
    await tools.call_tool("scraper.fetch_html", {"url": "https://x"})
    fake_mcp.call_tool.assert_called_once_with("scraper.fetch_html", {"url": "https://x"})
