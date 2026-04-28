from __future__ import annotations
from typing import Any
from llm4agents.transport.mcp import McpTransport
from llm4agents.tools.types import ToolDefinition


class _ScraperNamespace:
    def __init__(self, mcp: McpTransport) -> None:
        self._mcp = mcp

    async def fetch_html(self, url: str, **kwargs: Any) -> str:
        return await self._mcp.call_tool("scrape_url", {"url": url, **kwargs})

    async def markdown(self, url: str, **kwargs: Any) -> str:
        return await self._mcp.call_tool("scrape_markdown", {"url": url, **kwargs})

    async def links(self, url: str, **kwargs: Any) -> str:
        return await self._mcp.call_tool("scrape_links", {"url": url, **kwargs})

    async def screenshot(self, url: str, **kwargs: Any) -> str:
        return await self._mcp.call_tool("screenshot_url", {"url": url, **kwargs})

    async def pdf(self, url: str, **kwargs: Any) -> str:
        return await self._mcp.call_tool("pdf_url", {"url": url, **kwargs})

    async def extract(self, url: str, schema: dict[str, Any], **kwargs: Any) -> str:
        return await self._mcp.call_tool(
            "extract_structured", {"url": url, "schema": schema, **kwargs}
        )


class _SearchNamespace:
    def __init__(self, mcp: McpTransport) -> None:
        self._mcp = mcp

    async def google(self, query: str, **kwargs: Any) -> str:
        return await self._mcp.call_tool("google_search", {"query": query, **kwargs})

    async def google_batch(self, queries: list[str], **kwargs: Any) -> str:
        return await self._mcp.call_tool(
            "google_batch_search", {"queries": queries, **kwargs}
        )


class _ImageNamespace:
    def __init__(self, mcp: McpTransport) -> None:
        self._mcp = mcp

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self._mcp.call_tool("image_generate", {"prompt": prompt, **kwargs})

    async def edit(self, image: str, prompt: str, **kwargs: Any) -> str:
        return await self._mcp.call_tool(
            "image_edit", {"image": image, "prompt": prompt, **kwargs}
        )

    async def analyze(self, image: str, **kwargs: Any) -> str:
        return await self._mcp.call_tool("image_analyze", {"image": image, **kwargs})


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

    async def call(self, name: str, args: dict[str, Any]) -> str:
        return await self._mcp.call_tool(name, args)
