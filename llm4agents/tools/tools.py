from __future__ import annotations
from typing import Any
from llm4agents.transport.mcp import McpTransport, _normalize_content
from llm4agents.transport.mcp_stdio import McpServerHandle
from llm4agents.tools.connect import connect as _connect, McpServerConfig
from llm4agents.tools.types import ToolDefinition, McpToolResult, McpTextContent


class _ScraperNamespace:
    def __init__(self, mcp: McpTransport) -> None:
        self._mcp = mcp

    async def fetch_html(self, url: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("fetch_html", {"url": url, **kwargs})

    async def markdown(self, url: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("markdown", {"url": url, **kwargs})

    async def links(self, url: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("links", {"url": url, **kwargs})

    async def screenshot(self, url: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("screenshot", {"url": url, **kwargs})

    async def pdf(self, url: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("pdf", {"url": url, **kwargs})

    async def extract(self, url: str, schema: dict[str, Any], **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("extract", {"url": url, "schema": schema, **kwargs})

    async def session_create(self, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("session_create", {**kwargs})

    async def session_close(self, session_id: str) -> McpToolResult:
        return await self._mcp.call_tool("session_close", {"sessionId": session_id})

    async def session_status(self, session_id: str) -> McpToolResult:
        return await self._mcp.call_tool("session_status", {"sessionId": session_id})

    async def session_run(self, session_id: str, actions: list[Any], **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("session_exec", {"sessionId": session_id, "actions": actions, **kwargs})


class _SearchNamespace:
    def __init__(self, mcp: McpTransport) -> None:
        self._mcp = mcp

    async def google(self, q: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("google_search", {"q": q, **kwargs})

    async def google_news(self, q: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("google_news", {"q": q, **kwargs})

    async def google_maps(self, q: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("google_maps", {"q": q, **kwargs})

    async def google_batch(self, queries: list[str], **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("google_batch_search", {"queries": queries, **kwargs})


class _ImageNamespace:
    def __init__(self, mcp: McpTransport) -> None:
        self._mcp = mcp

    async def generate(self, prompt: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("generate_image", {"prompt": prompt, **kwargs})

    async def edit(self, prompt: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("edit_image", {"prompt": prompt, **kwargs})

    async def analyze(self, prompt: str, **kwargs: Any) -> McpToolResult:
        return await self._mcp.call_tool("analyze_image", {"prompt": prompt, **kwargs})


class Tools:
    def __init__(self, mcp: McpTransport) -> None:
        self._mcp = mcp
        self._definitions_cache: list[ToolDefinition] | None = None
        self._servers: dict[str, McpServerHandle] = {}
        self.scraper = _ScraperNamespace(mcp)
        self.search = _SearchNamespace(mcp)
        self.image = _ImageNamespace(mcp)

    @property
    def definitions(self) -> list[ToolDefinition]:
        return self._definitions_cache or []

    async def fetch_definitions(self) -> list[ToolDefinition]:
        raw = await self._mcp.list_tools()
        defs = [
            ToolDefinition(
                name=t["name"],
                description=t.get("description", ""),
                inputSchema=t.get("inputSchema", {}),
            )
            for t in raw
        ]
        self._definitions_cache = defs
        return defs

    async def connect(self, cfg: McpServerConfig) -> McpServerHandle:
        if cfg.name in self._servers:
            raise ValueError(f'mcp server "{cfg.name}" is already connected')
        handle = await _connect(cfg)
        self._servers[cfg.name] = handle
        return handle

    async def disconnect(self, name: str) -> None:
        """Disconnect a registered MCP server by name.

        Raises ``ValueError`` if no server with that name is connected — matches
        the TS SDK's ``Tools.disconnect()`` behavior for cross-language parity.
        """
        handle = self._servers.pop(name, None)
        if handle is None:
            raise ValueError(f'No connected MCP server named "{name}".')
        await handle.disconnect()

    async def disconnect_all(self) -> None:
        """Disconnect all registered MCP servers and clear the registry.

        Best-effort: every handle's ``disconnect()`` is dispatched even if
        some fail. Errors are collected and re-raised as an aggregate after
        all disconnects have settled, so a single broken handle can't leak
        the rest of the registry — each underlying child process / HTTP
        transport still gets the close signal. The registry is cleared in
        all cases (including on partial failure).
        """
        import asyncio

        handles = list(self._servers.values())
        self._servers.clear()
        results = await asyncio.gather(
            *(h.disconnect() for h in handles), return_exceptions=True
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            # ExceptionGroup is Python 3.11+; fall back to RuntimeError on 3.10.
            if hasattr(__builtins__, "ExceptionGroup") or "ExceptionGroup" in dir(__builtins__):
                raise ExceptionGroup(  # type: ignore[name-defined]  # noqa: F821
                    f"disconnect_all: {len(errors)} handle(s) failed", errors
                )
            joined = "; ".join(str(e) for e in errors)
            raise RuntimeError(
                f"disconnect_all: {len(errors)} handle(s) failed: {joined}"
            ) from errors[0]

    def connected_servers(self) -> list[str]:
        return list(self._servers.keys())

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        server: str | None = None,
    ) -> McpToolResult:
        if server is not None:
            handle = self._servers.get(server)
            if handle is None:
                raise ValueError(f'mcp server "{server}" is not connected')
            raw = await handle.call_tool(name, args)
            raw_content: list[dict[str, Any]] = raw.get("content", [])
            content = tuple(_normalize_content(item) for item in raw_content)
            text = "\n".join(c.text for c in content if isinstance(c, McpTextContent))
            return McpToolResult(content=content, text=text)
        # Existing proxy-gateway path
        return await self._mcp.call_tool(name, args)

    async def call(self, name: str, args: dict[str, Any]) -> McpToolResult:
        return await self._mcp.call_tool(name, args)
