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
async def test_workspace_create(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"workspace_id": "ws-123"}'))
    result = await tools.workspace.create()
    assert "ws-123" in result.text


@respx.mock
async def test_workspace_list_no_params(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('["file1.txt", "file2.txt"]'))
    result = await tools.workspace.list()
    assert "file1.txt" in result.text


@respx.mock
async def test_workspace_list_with_params(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('["images/cat.png"]'))
    result = await tools.workspace.list(prefix="images/", limit=10)
    assert "cat.png" in result.text


@respx.mock
async def test_workspace_stat(tools):
    respx.post("https://mcp.example.com/mcp").mock(
        return_value=_mcp_text_response('{"filename": "report.pdf", "size_bytes": 4096}')
    )
    result = await tools.workspace.stat("report.pdf")
    assert "report.pdf" in result.text


@respx.mock
async def test_workspace_delete(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"deleted": true}'))
    result = await tools.workspace.delete("old_file.txt")
    assert "deleted" in result.text


@respx.mock
async def test_workspace_upload(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"url": "https://cdn.example.com/file.txt"}'))
    result = await tools.workspace.upload(
        filename="file.txt",
        content_base64="aGVsbG8=",
        days_to_store=7,
    )
    assert "url" in result.text


@respx.mock
async def test_workspace_upload_with_content_type(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"url": "https://cdn.example.com/img.png"}'))
    result = await tools.workspace.upload(
        filename="img.png",
        content_base64="iVBOR==",
        days_to_store=3,
        content_type="image/png",
    )
    assert "url" in result.text


@respx.mock
async def test_workspace_upload_init(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"upload_id": "up-abc123"}'))
    result = await tools.workspace.upload_init(
        filename="large.bin",
        size_bytes=10_485_760,
        days_to_store=14,
    )
    assert "upload_id" in result.text


@respx.mock
async def test_workspace_upload_finalize(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"status": "complete"}'))
    result = await tools.workspace.upload_finalize("up-abc123")
    assert "complete" in result.text


@respx.mock
async def test_workspace_download(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"url": "https://cdn.example.com/report.pdf?token=xyz"}'))
    result = await tools.workspace.download("report.pdf")
    assert "url" in result.text


@respx.mock
async def test_workspace_download_with_options(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"url": "https://cdn.example.com/report.pdf?token=xyz"}'))
    result = await tools.workspace.download("report.pdf", format="url", url_ttl_minutes=30)
    assert "url" in result.text


@respx.mock
async def test_workspace_extend(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"expires_at": "2026-06-13"}'))
    result = await tools.workspace.extend("report.pdf", additional_days=14)
    assert "expires_at" in result.text


@respx.mock
async def test_workspace_copy(tools):
    respx.post("https://mcp.example.com/mcp").mock(return_value=_mcp_text_response('{"filename": "backup/report.pdf"}'))
    result = await tools.workspace.copy(
        source_filename="report.pdf",
        dest_filename="backup/report.pdf",
        days_to_store=30,
    )
    assert "backup" in result.text
