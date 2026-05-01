from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from llm4agents.transport.http import HttpTransport
from llm4agents.transport.mcp import McpTransport
from llm4agents.chat.completions import ChatCompletions
from llm4agents.chat.conversation import Conversation
from llm4agents.wallets.wallets import Wallets
from llm4agents.transfer.transfer import Transfer
from llm4agents.tools.tools import Tools
from llm4agents.agents import Agents

_DEFAULT_BASE_URL = "https://api.llm4agents.com"
_DEFAULT_MCP_URL = "https://mcp.llm4agents.com/mcp"
_DEFAULT_TIMEOUT = 30.0
_MCP_TIMEOUT = 60.0


@dataclass(frozen=True)
class ModelListResult:
    models: list[dict[str, Any]]
    request_id: str | None
    fee_pct: int | None = None


class _ChatNamespace:
    def __init__(self, completions: ChatCompletions, http: HttpTransport) -> None:
        self.completions = completions
        self._http = http

    def conversation(self, opts: dict[str, Any]) -> Conversation:
        return Conversation(self._http, opts)


class _ModelsNamespace:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def list(self, search: str | None = None) -> ModelListResult:
        params: dict[str, str] | None = {"search": search} if search else None
        data = await self._http.get("/api/v1/models", params=params)
        return ModelListResult(
            models=data.get("models", []),
            fee_pct=data.get("feePct"),
            request_id=data.get("requestId"),
        )


class LLM4AgentsClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        mcp_url: str = _DEFAULT_MCP_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._http = HttpTransport(base_url, api_key, timeout)
        mcp = McpTransport(mcp_url, api_key, _MCP_TIMEOUT)

        completions = ChatCompletions(self._http)
        self.chat = _ChatNamespace(completions, self._http)
        self.wallets = Wallets(self._http)
        self.transfer = Transfer(self._http)
        self.tools = Tools(mcp)
        self.models = _ModelsNamespace(self._http)
        self.agents = Agents(self._http)
