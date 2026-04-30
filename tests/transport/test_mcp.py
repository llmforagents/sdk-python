import pytest
import respx
import httpx
from llm4agents.transport.mcp import McpTransport, sniff_mime_type
from llm4agents.tools.types import McpImageContent, McpResourceContent, McpTextContent
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


# --- New tests for normalization ---

@respx.mock
async def test_json_in_text_image_base64(mcp):
    """text block containing {"imageBase64":"iVBOR..."} → McpImageContent."""
    png_b64 = "iVBORw0KGgo="  # starts with iVBO → image/png
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=httpx.Response(200, json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": f'{{"imageBase64":"{png_b64}"}}'}
                ]
            }
        })
    )
    result = await mcp.call_tool("capture_screen", {})
    assert len(result.content) == 1
    item = result.content[0]
    assert isinstance(item, McpImageContent)
    assert item.data == png_b64
    assert item.mimeType == "image/png"


@respx.mock
async def test_json_in_text_wrapper(mcp):
    """text block containing {"text":"hello","costCents":5} → McpTextContent("hello")."""
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=httpx.Response(200, json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": '{"text":"hello world","costCents":5}'}
                ]
            }
        })
    )
    result = await mcp.call_tool("query_tool", {})
    assert len(result.content) == 1
    item = result.content[0]
    assert isinstance(item, McpTextContent)
    assert item.text == "hello world"
    assert result.text == "hello world"


@respx.mock
async def test_field_aliasing_mime_type(mcp):
    """image block with mime_type (snake_case) → McpImageContent with correct mimeType."""
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=httpx.Response(200, json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "image",
                        "data": "/9j/abc=",  # starts with /9j/ → image/jpeg, but overridden by mime_type
                        "mime_type": "image/png",
                    }
                ]
            }
        })
    )
    result = await mcp.call_tool("screenshot", {})
    assert len(result.content) == 1
    item = result.content[0]
    assert isinstance(item, McpImageContent)
    assert item.mimeType == "image/png"
    assert item.data == "/9j/abc="


@respx.mock
async def test_json_in_text_pdf_base64(mcp):
    """text block with {"pdfBase64":"JVBE..."} → McpResourceContent with mimeType=application/pdf."""
    pdf_b64 = "JVBERi0xLjQ="  # starts with JVBE → application/pdf
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=httpx.Response(200, json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": f'{{"pdfBase64":"{pdf_b64}"}}'}
                ]
            }
        })
    )
    result = await mcp.call_tool("export_pdf", {})
    assert len(result.content) == 1
    item = result.content[0]
    assert isinstance(item, McpResourceContent)
    assert item.mimeType == "application/pdf"
    assert item.text == pdf_b64


# sniff_mime_type unit tests

def test_sniff_mime_type_png():
    assert sniff_mime_type("iVBORw0KGgo=") == "image/png"


def test_sniff_mime_type_jpeg():
    assert sniff_mime_type("/9j/abc") == "image/jpeg"


def test_sniff_mime_type_pdf():
    assert sniff_mime_type("JVBERi0=") == "application/pdf"


def test_sniff_mime_type_gif():
    assert sniff_mime_type("R0lGODlh") == "image/gif"


def test_sniff_mime_type_webp():
    assert sniff_mime_type("UklGRg==") == "image/webp"


def test_sniff_mime_type_default():
    assert sniff_mime_type("AAAA") == "image/png"
