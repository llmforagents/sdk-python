from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from llm4agents.errors import LLM4AgentsError, ErrorCode
from llm4agents.wallets.types import WalletInfo, Balance, WalletBalance, Transaction, TransactionList
from llm4agents.transfer.types import QuoteResult, TransferResult
from llm4agents.chat.types import ChatMessage, ChatResponse, StreamChunk, ResponseMeta
from llm4agents.chat.conversation import ConversationResponse
from llm4agents.tools.types import McpToolResult, McpTextContent, McpImageContent, McpResourceContent, McpContent


@dataclass(frozen=True)
class ModelListResult:
    models: list[dict[str, Any]]
    request_id: str | None


__all__ = [
    "LLM4AgentsError",
    "ErrorCode",
    "WalletInfo",
    "Balance",
    "WalletBalance",
    "Transaction",
    "TransactionList",
    "QuoteResult",
    "TransferResult",
    "ChatMessage",
    "ChatResponse",
    "StreamChunk",
    "ResponseMeta",
    "ConversationResponse",
    "McpToolResult",
    "McpTextContent",
    "McpImageContent",
    "McpResourceContent",
    "McpContent",
    "ModelListResult",
]
