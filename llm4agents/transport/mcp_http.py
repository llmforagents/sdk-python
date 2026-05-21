from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from .mcp_stdio import McpServerHandle  # reuse the handle


@dataclass(frozen=True)
class HttpServerConfig:
    name: str
    url: str
    headers: dict[str, str] | None = None


async def connect_http(
    cfg: HttpServerConfig,
    connect_timeout_s: float = 5.0,
) -> McpServerHandle:
    """Connect to an MCP server over Streamable HTTP.

    Uses the non-deprecated ``streamable_http_client`` from mcp>=1.x.
    Headers are injected via a pre-configured ``httpx.AsyncClient``.
    The returned ``McpServerHandle`` owns the connection lifecycle; call
    ``handle.disconnect()`` when done.
    """
    exit_stack = AsyncExitStack()
    try:
        http_client = create_mcp_http_client(
            headers=cfg.headers,
            timeout=httpx.Timeout(connect_timeout_s, read=300.0),
        )
        await exit_stack.enter_async_context(http_client)
        streams = await asyncio.wait_for(
            exit_stack.enter_async_context(
                streamable_http_client(cfg.url, http_client=http_client)
            ),
            timeout=connect_timeout_s,
        )
        # streamable_http_client yields (read_stream, write_stream, get_session_id)
        session = await exit_stack.enter_async_context(
            ClientSession(streams[0], streams[1])
        )
        await asyncio.wait_for(session.initialize(), timeout=connect_timeout_s)
        return McpServerHandle(cfg.name, session, exit_stack)
    except Exception:
        await exit_stack.aclose()
        raise
