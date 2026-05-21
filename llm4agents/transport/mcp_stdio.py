from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Sequence

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@dataclass(frozen=True)
class StdioServerConfig:
    name: str
    command: str
    args: Sequence[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None


class McpServerHandle:
    """Lifecycle-owning handle returned by connect_stdio / connect_http.

    The SDK does NOT auto-reconnect when a stdio server's child process dies.
    Callers detecting failure (via call_tool errors) must call connect_stdio()
    again with the original config to recover.
    """

    def __init__(self, name: str, session: ClientSession, exit_stack: AsyncExitStack) -> None:
        self._name = name
        self._session = session
        self._exit_stack = exit_stack

    @property
    def name(self) -> str:
        return self._name

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = await self._session.call_tool(tool_name, args)
        return {
            "content": [
                {"type": c.type, "text": getattr(c, "text", None)}
                for c in result.content
            ]
        }

    async def disconnect(self) -> None:
        await self._exit_stack.aclose()


async def connect_stdio(
    cfg: StdioServerConfig,
    connect_timeout_s: float = 5.0,
) -> McpServerHandle:
    exit_stack = AsyncExitStack()
    params = StdioServerParameters(
        command=cfg.command,
        args=list(cfg.args),
        env=cfg.env,
        cwd=cfg.cwd,
    )
    try:
        streams = await asyncio.wait_for(
            exit_stack.enter_async_context(stdio_client(params)),
            timeout=connect_timeout_s,
        )
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        await asyncio.wait_for(session.initialize(), timeout=connect_timeout_s)
        return McpServerHandle(cfg.name, session, exit_stack)
    except Exception:
        # Clean up partially-opened resources on failure
        await exit_stack.aclose()
        raise
