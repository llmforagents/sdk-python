# llm4agents-sdk-python Project Overview

## Purpose
Python SDK for llm4agents.com — gasless AI agent infrastructure. Provides client library for interacting with the LLM4Agents API.

## Tech Stack
- **Language:** Python 3.10+
- **HTTP:** httpx (async with HTTP/2)
- **Web3:** eth-account (for wallet signing)
- **Testing:** pytest, pytest-asyncio, respx (HTTP mocking)
- **Build:** hatchling

## Architecture Overview
- **Entry Point:** `llm4agents/client.py` - Main LLM4AgentsClient facade
- **Transport:** `llm4agents/transport/` - HTTP and MCP transports
- **Modules:**
  - `llm4agents/chat/` - Chat completions and conversations
  - `llm4agents/wallets/` - Wallet management
  - `llm4agents/transfer/` - Token transfers with signing
  - `llm4agents/tools/` - MCP tool integrations (scraper, search, image)
- **Error Handling:** `llm4agents/errors.py` - Typed error codes and mapping

## Module Structure
- All async-first design (httpx AsyncClient)
- Type hints throughout
- Namespace pattern for client submodules (e.g., client.chat, client.wallets)
- Conversation factory pattern for interactive chat
- Tool call support with callback hooks

## Key Files to Know
- `llm4agents/__init__.py` - Package exports (currently empty)
- `llm4agents/types.py` - Type re-exports (to be created)
- `llm4agents/client.py` - Main client facade (to be created)
- Tests in `tests/` with structure matching source modules

## Code Style
- Type hints required on all functions
- Use dataclass with frozen=True for immutable types
- Async/await for I/O operations
- Error handling via LLM4AgentsError with typed codes
- Namespace classes for logical grouping
