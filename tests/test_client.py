import pytest
from llm4agents import LLM4AgentsClient, LLM4AgentsError
from llm4agents.wallets.types import WalletInfo, Balance, Transaction, TransactionList
from llm4agents.transfer.types import QuoteResult, TransferResult
from llm4agents.chat.types import ChatResponse, StreamChunk
from llm4agents.chat.conversation import Conversation, ConversationResponse


def test_client_creates():
    client = LLM4AgentsClient(api_key="sk-test")
    assert hasattr(client, "chat")
    assert hasattr(client, "wallets")
    assert hasattr(client, "transfer")
    assert hasattr(client, "tools")
    assert hasattr(client, "models")


def test_chat_namespace():
    client = LLM4AgentsClient(api_key="sk-test")
    assert hasattr(client.chat, "completions")
    assert callable(client.chat.conversation)


def test_conversation_factory():
    client = LLM4AgentsClient(api_key="sk-test")
    conv = client.chat.conversation({"model": "gpt-4o"})
    assert isinstance(conv, Conversation)


def test_error_is_exported():
    from llm4agents import LLM4AgentsError
    err = LLM4AgentsError("test", "api_error", 500, None)
    assert err.code == "api_error"


def test_custom_base_url():
    client = LLM4AgentsClient(api_key="sk-test", base_url="https://custom.example.com")
    assert client._http._base_url == "https://custom.example.com"
