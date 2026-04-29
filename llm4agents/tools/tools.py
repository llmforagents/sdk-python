from __future__ import annotations
from typing import Any
from llm4agents.transport.mcp import McpTransport
from llm4agents.tools.types import ToolDefinition, McpToolResult


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

    async def call(self, name: str, args: dict[str, Any]) -> McpToolResult:
        return await self._mcp.call_tool(name, args)
