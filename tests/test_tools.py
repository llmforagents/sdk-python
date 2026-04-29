import pytest
import respx
import httpx
from llm4agents.transport.mcp import McpTransport
from llm4agents.tools.tools import Tools


@pytest.fixture
def mcp():
    return McpTransport("https://mcp.example.com/mcp", "test-key", 10.0)


@pytest.fixture
def tools(mcp):
    return Tools(mcp)


def _mcp_text_response(text: str) -> httpx.Response:
    return httpx.Response(200, json={
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": text}]},
    })


@respx.mock
async def test_fetch_definitions(tools):
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {
                            "name": "fetch_html",
                            "description": "Fetch HTML",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"url": {"type": "string"}},
                            },
                        },
                        {
                            "name": "google_search",
                            "description": "Google search",
                            "inputSchema": {},
                        },
                    ]
                },
            },
        )
    )
    defs = await tools.fetch_definitions()
    assert len(defs) == 2
    assert defs[0]["name"] == "fetch_html"
    assert len(tools.definitions) == 2


@respx.mock
async def test_call(tools):
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=_mcp_text_response("<html>page</html>")
    )
    result = await tools.call("fetch_html", {"url": "https://example.com"})
    assert result.text == "<html>page</html>"


@respx.mock
async def test_scraper_fetch_html(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response("<html>ok</html>"))
    result = await tools.scraper.fetch_html("https://example.com")
    assert "<html>" in result.text


@respx.mock
async def test_scraper_markdown(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response("# Title\nContent"))
    result = await tools.scraper.markdown("https://example.com")
    assert "# Title" in result.text


@respx.mock
async def test_scraper_links(tools):
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=_mcp_text_response("https://example.com/1\nhttps://example.com/2")
    )
    result = await tools.scraper.links("https://example.com")
    assert "https://example.com/1" in result.text


@respx.mock
async def test_scraper_screenshot(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response("data:image/png;base64,..."))
    result = await tools.scraper.screenshot("https://example.com")
    assert "data:image" in result.text


@respx.mock
async def test_scraper_pdf(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response("PDF content here"))
    result = await tools.scraper.pdf("https://example.com")
    assert "PDF" in result.text


@respx.mock
async def test_scraper_extract(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"title": "Example"}'))
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    result = await tools.scraper.extract("https://example.com", schema)
    assert "title" in result.text


@respx.mock
async def test_search_google(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response("Search results"))
    result = await tools.search.google("python")
    assert "Search" in result.text


@respx.mock
async def test_search_google_batch(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response("Batch results"))
    result = await tools.search.google_batch(["python", "golang"])
    assert "Batch" in result.text


@respx.mock
async def test_image_generate(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response("data:image/png;base64,..."))
    result = await tools.image.generate("a cat")
    assert "data:image" in result.text


@respx.mock
async def test_image_edit(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response("data:image/png;base64,..."))
    result = await tools.image.edit("make it blue")
    assert "data:image" in result.text


@respx.mock
async def test_image_analyze(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response("The image shows..."))
    result = await tools.image.analyze("What is this?")
    assert "image" in result.text
