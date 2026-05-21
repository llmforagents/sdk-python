from __future__ import annotations

from typing import Union

from llm4agents.transport.mcp_stdio import StdioServerConfig, McpServerHandle, connect_stdio
from llm4agents.transport.mcp_http import HttpServerConfig, connect_http

McpServerConfig = Union[StdioServerConfig, HttpServerConfig]


async def connect(
    cfg: McpServerConfig,
    connect_timeout_s: float = 5.0,
) -> McpServerHandle:
    """Route to stdio or http connect based on config shape."""
    if isinstance(cfg, HttpServerConfig):
        return await connect_http(cfg, connect_timeout_s=connect_timeout_s)
    return await connect_stdio(cfg, connect_timeout_s=connect_timeout_s)
