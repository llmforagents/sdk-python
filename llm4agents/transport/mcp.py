from __future__ import annotations
from typing import Any, TYPE_CHECKING
import json
import httpx
from llm4agents.errors import LLM4AgentsError, map_http_error

if TYPE_CHECKING:
    from llm4agents.tools.types import McpToolResult


def sniff_mime_type(base64: str) -> str:
    """Sniff MIME type from the first 4 characters of a base64-encoded payload."""
    prefix = base64[:4]
    if prefix == "iVBO":
        return "image/png"
    if prefix == "/9j/":
        return "image/jpeg"
    if prefix == "JVBE":
        return "application/pdf"
    if prefix == "R0lG":
        return "image/gif"
    if prefix == "UklG":
        return "image/webp"
    return "image/png"


def _extract_text(c: dict[str, Any]) -> str:
    """Extract text from a content block, unwrapping JSON wrappers like {"text":"..."}."""
    raw = c.get("text", "")
    if not isinstance(raw, str):
        if isinstance(raw, dict):
            candidate = raw.get("text")
            if isinstance(candidate, str):
                return candidate
        return ""
    # Check for JSON wrapper {"text": "...", ...}
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                return parsed["text"]
        except (json.JSONDecodeError, ValueError):
            pass
    return raw


def _normalize_content(c: dict[str, Any]) -> Any:
    """Normalize a raw MCP content block into a typed McpContent.

    Handles:
    - Field aliasing: mime_type -> mimeType, imageBase64/pngBase64 -> data
    - JSON-in-text unwrap: text blocks containing {"imageBase64":...}, {"pdfBase64":...},
      {"pngBase64":...} are promoted to image/resource content.
    """
    # Lazy import to avoid circular dependency
    from llm4agents.tools.types import McpImageContent, McpResourceContent, McpTextContent  # noqa: PLC0415

    t = c.get("type", "text")

    if t == "image":
        data = (
            c.get("data")
            or c.get("imageBase64")
            or c.get("pngBase64")
            or c.get("pdfBase64")
            or ""
        )
        mime_type: str = (
            c.get("mimeType")
            or c.get("mime_type")
            or sniff_mime_type(str(data))
        )
        return McpImageContent(type="image", data=str(data), mimeType=mime_type)

    if t == "resource":
        mime_type_res: str | None = c.get("mimeType") or c.get("mime_type")
        return McpResourceContent(
            type="resource",
            uri=c.get("uri", ""),
            text=c.get("text"),
            mimeType=mime_type_res,
        )

    # default: text — but first try to unwrap JSON payloads
    text_value = _extract_text(c)

    # Try to detect JSON-wrapped image/pdf payloads embedded in a text block
    stripped = text_value.lstrip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(text_value)
            if isinstance(parsed, dict):
                base64_val = (
                    parsed.get("imageBase64")
                    or parsed.get("pngBase64")
                    or parsed.get("pdfBase64")
                )
                if base64_val is not None and isinstance(base64_val, str):
                    resolved_mime: str = (
                        parsed.get("mimeType")
                        or parsed.get("mime_type")
                        or sniff_mime_type(base64_val)
                    )
                    # PDF wrapped in text block
                    if resolved_mime == "application/pdf" or "pdfBase64" in parsed:
                        return McpResourceContent(
                            type="resource",
                            uri="",
                            text=base64_val,
                            mimeType="application/pdf",
                        )
                    return McpImageContent(type="image", data=base64_val, mimeType=resolved_mime)
        except (json.JSONDecodeError, ValueError):
            pass

    return McpTextContent(type="text", text=text_value)


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
        from llm4agents.tools.types import McpTextContent, McpToolResult  # noqa: PLC0415

        result = await self._rpc("tools/call", {"name": name, "arguments": args})
        raw_content: list[dict[str, Any]] = result.get("content", [])

        if result.get("isError"):
            err_text = "\n".join(
                _extract_text(item) for item in raw_content if item.get("type") == "text"
            )
            raise LLM4AgentsError(
                err_text or f"Tool {name} failed", "tool_execution_error", None, None
            )

        content = [_normalize_content(item) for item in raw_content]

        text = "\n".join(c.text for c in content if isinstance(c, McpTextContent))
        return McpToolResult(content=tuple(content), text=text)
