from __future__ import annotations
from typing import Any, TYPE_CHECKING
import json
import httpx
from llm4agents.errors import LLM4AgentsError, map_http_error

if TYPE_CHECKING:
    from llm4agents.tools.types import McpToolResult


class McpTransport:
    def __init__(self, mcp_url: str, api_key: str, timeout: float) -> None:
        self._url = mcp_url
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout
        self._tools_cache: list[dict[str, Any]] | None = None
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                res = await client.post(
                    self._url,
                    content=json.dumps(payload),
                    headers=self._headers,
                )
            except httpx.TimeoutException as e:
                raise LLM4AgentsError(str(e), "timeout", None, None) from e
            except httpx.NetworkError as e:
                raise LLM4AgentsError(str(e), "network_error", None, None) from e
            if res.status_code >= 400:
                raise map_http_error(res.status_code, res.json(), None)
            data = res.json()
            if "error" in data:
                rpc_err = data["error"]
                code_int: int = rpc_err.get("code", 0)
                msg: str = rpc_err.get("message", "MCP error")
                sdk_code = "tool_not_found" if code_int == -32601 else "tool_execution_error"
                raise LLM4AgentsError(msg, sdk_code, None, None)
            return data.get("result")

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._tools_cache is not None:
            return self._tools_cache
        result = await self._rpc("tools/list", {})
        tools: list[dict[str, Any]] = result.get("tools", [])
        self._tools_cache = tools
        return tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> McpToolResult:
        # Lazy import to avoid circular dependency: mcp → tools.types → tools.__init__ → tools.tools → mcp
        from llm4agents.tools.types import McpImageContent, McpResourceContent, McpTextContent, McpToolResult  # noqa: PLC0415

        result = await self._rpc("tools/call", {"name": name, "arguments": args})
        raw_content: list[dict[str, Any]] = result.get("content", [])

        if result.get("isError"):
            err_text = "\n".join(
                item.get("text", "") for item in raw_content if item.get("type") == "text"
            )
            raise LLM4AgentsError(
                err_text or f"Tool {name} failed", "tool_execution_error", None, None
            )

        content = []
        for item in raw_content:
            t = item.get("type", "text")
            if t == "image":
                content.append(McpImageContent(type="image", data=item.get("data", ""), mimeType=item.get("mimeType", "")))
            elif t == "resource":
                content.append(McpResourceContent(type="resource", uri=item.get("uri", ""), text=item.get("text"), mimeType=item.get("mimeType")))
            else:
                content.append(McpTextContent(type="text", text=item.get("text", "")))

        text = "\n".join(c.text for c in content if isinstance(c, McpTextContent))
        return McpToolResult(content=tuple(content), text=text)
