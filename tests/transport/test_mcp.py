import pytest
import respx
import httpx
from llm4agents.transport.mcp import McpTransport
from llm4agents.errors import LLM4AgentsError


@pytest.fixture
def mcp():
    return McpTransport("https://mcp.example.com/mcp", "test-key", 10.0)


@respx.mock
async def test_list_tools(mcp):
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=httpx.Response(200, json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [{"name": "scrape_url", "description": "Scrape a URL", "inputSchema": {}}]
            }
        })
    )
    tools = await mcp.list_tools()
    assert tools[0]["name"] == "scrape_url"
    # Second call uses cache
    tools2 = await mcp.list_tools()
    assert tools2 == tools


@respx.mock
async def test_call_tool(mcp):
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=httpx.Response(200, json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "Hello world"}]}
        })
    )
    result = await mcp.call_tool("scrape_url", {"url": "https://example.com"})
    assert result.text == "Hello world"
    assert len(result.content) == 1


@respx.mock
async def test_call_tool_error(mcp):
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=httpx.Response(200, json={
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"}
        })
    )
    with pytest.raises(LLM4AgentsError) as exc_info:
        await mcp.call_tool("unknown_tool", {})
    assert exc_info.value.code == "tool_not_found"
