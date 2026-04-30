# llm4agents-sdk

[![PyPI version](https://img.shields.io/pypi/v/llm4agents-sdk)](https://pypi.org/project/llm4agents-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/llm4agents-sdk)](https://pypi.org/project/llm4agents-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python SDK for [llm4agents.com](https://llm4agents.com) — gasless AI agent infrastructure. Chat completions, wallet management, gasless stablecoin transfers, and MCP-powered tools through a single async client.

## Install

```bash
pip install llm4agents-sdk
```

`eth-account` is bundled as a dependency for gasless transfers (no extra install needed).

## Get an API Key

1. Go to **[api.llm4agents.com/docs](https://api.llm4agents.com/docs)**
2. Register your agent to receive a key in the format `sk-proxy-...`
3. Pass it to the client constructor

## Quick Start

```python
import asyncio
from llm4agents import LLM4AgentsClient

async def main():
    client = LLM4AgentsClient(api_key="sk-proxy-...")

    # Chat completion
    response = await client.chat.completions.create({
        "model": "anthropic/claude-sonnet-4",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    print(response["choices"][0]["message"]["content"])

    # Conversation with MCP tools
    conv = client.chat.conversation({
        "model": "anthropic/claude-sonnet-4",
        "system": "You are a research assistant",
        "tools": client.tools,
    })
    answer = await conv.say("Search for Bitcoin news and summarize the top 3")
    print(answer.content)

    # Gasless stablecoin transfer
    result = await client.transfer.send({
        "chain": "polygon", "token": "USDC",
        "to": "0xRecipient...", "amount": "10.50",
        "private_key": "0x...",
    })
    print(result["tx_hash"], result["explorer_url"])

asyncio.run(main())
```

## Chat

### Completions

```python
# Non-streaming
response = await client.chat.completions.create({
    "model": "anthropic/claude-sonnet-4",
    "messages": [{"role": "user", "content": "Hello"}],
})
print(response["choices"][0]["message"]["content"])

# Streaming
stream = await client.chat.completions.create({
    "model": "anthropic/claude-sonnet-4",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": True,
})
async for chunk in stream:
    print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)

# With extended thinking
response = await client.chat.completions.create({
    "model": "anthropic/claude-sonnet-4",
    "messages": [{"role": "user", "content": "Solve step by step: 47 * 83"}],
    "reasoning": True,
    "include_reasoning": True,
})

# Force/restrict tool use
response = await client.chat.completions.create({
    "model": "openai/gpt-4o",
    "messages": [...],
    "tools": await client.tools.fetch_definitions(),
    "tool_choice": "required",  # "none" | "auto" | "required" | {"type": "function", "function": {"name": "..."}}
})

```

### Conversation with Tools

```python
from llm4agents.errors import LLM4AgentsError

conv = client.chat.conversation({
    "model": "anthropic/claude-sonnet-4",
    "system": "You are a research assistant",
    "tools": client.tools,
    "history": [],          # optional: rehydrate from persisted messages
    "on_tool_call": lambda name, args: print(f"Calling {name}...") or True,
    "on_tool_result": lambda name, result: print(f"{name}: {result.text[:80]}"),
    "max_tool_rounds": 5,   # default 10
})

# Single turn
answer = await conv.say("Search for Bitcoin news and summarize the top 3")
print(answer.content)
for call in answer.tool_calls:
    print(f"  {call['name']} → {call['result'].text[:60]}")

# History management
messages = conv.messages    # list[ChatMessage] — JSON-serializable
conv.clear()                # reset to empty, keeps system prompt and options
branch = conv.fork()        # copy history into a new Conversation
```

`on_tool_call` receives `(name: str, args: dict)` and should return `True` to proceed or `False` to cancel the tool call.

To restore a conversation from a previous session:

```python
conv = client.chat.conversation({
    "model": "anthropic/claude-sonnet-4",
    "history": saved_messages,   # rehydrate from your store
})
```

### McpToolResult

All tool calls return an `McpToolResult` instead of a plain string:

```python
result = await client.tools.scraper.fetch_html("https://example.com")

result.text          # str — joined text from all text parts (convenience)
str(result)          # same as result.text — backward-compatible
result.content       # tuple[McpContent, ...] — typed content parts

for part in result.content:
    if part.type == "text":
        print(part.text)
    elif part.type == "image":
        print(part.mime_type, len(part.data))   # base64-encoded image
    elif part.type == "resource":
        print(part.uri, part.text)
```

## Wallets

```python
# Generate a deposit wallet
wallet = await client.wallets.generate({"chain": "polygon", "token": "USDC"})
print(wallet["address"])

# Check balance
balance = await client.wallets.balance()
print(balance["available_usd"])
for w in balance["wallets"]:
    print(f'{w["chain"]}/{w["token"]}: ${w["available_usd"]}')

# Transaction history
txs = await client.wallets.transactions({"limit": 20, "type": "deposit"})
for tx in txs["transactions"]:
    print(f'{tx["type"]}: ${tx["amount_usd_cents"] / 100} — {tx["description"]}')
```

`type` filter accepts `"deposit"`, `"usage"`, or `"refund"`.

## Gasless Transfers

```python
# One-call convenience
result = await client.transfer.send({
    "chain": "polygon", "token": "USDC",
    "to": "0xRecipient...", "amount": "10.50",
    "private_key": "0x...",
})
print(result["tx_hash"], result["explorer_url"])

# Two-step — inspect the fee before committing
quote = await client.transfer.quote({
    "chain": "polygon", "token": "USDC",
    "from_": "0xSender...", "to": "0xRecipient...", "amount": "10.50",
})
print(f"Fee: {quote['fee_formatted']}")

result = await client.transfer.submit(quote, "0xPrivateKey...")
print(result["tx_hash"])
```

## MCP Tools

All tool methods return `McpToolResult`. Access `.text` for the plain-text representation.

### Scraper

```python
html   = await client.tools.scraper.fetch_html("https://example.com")
md     = await client.tools.scraper.markdown("https://example.com")
links  = await client.tools.scraper.links("https://example.com")
shot   = await client.tools.scraper.screenshot("https://example.com", full_page=True)
pdf    = await client.tools.scraper.pdf("https://example.com")
data   = await client.tools.scraper.extract("https://example.com", schema={
    "type": "object",
    "properties": {"title": {"type": "string"}},
})
```

All scraper methods accept an optional `proxy` keyword: `"none"`, `"datacenter"`, or `"residential"`.

#### Browser sessions

```python
session = await client.tools.scraper.session_create(proxy="residential")
result  = await client.tools.scraper.session_run(
    session_id=session.text,
    actions=[{"type": "navigate", "url": "https://example.com"}],
)
status  = await client.tools.scraper.session_status(session_id=session.text)
await client.tools.scraper.session_close(session_id=session.text)
```

### Search

```python
results = await client.tools.search.google("TypeScript SDK design")
news    = await client.tools.search.google_news("Bitcoin", tbs="qdr:d")
places  = await client.tools.search.google_maps("coffee near me")
batch   = await client.tools.search.google_batch(["python", "golang"])
```

### Image

```python
img      = await client.tools.image.generate("A robot writing code")
edited   = await client.tools.image.edit("Make it blue", image_url="https://...")
analysis = await client.tools.image.analyze("What is this?", image_url="https://...")
```

`image.generate` and `image.edit` may return `McpImageContent` parts (base64 PNG) alongside text.

### Tool Definitions

`client.tools.definitions` returns the cached `list[ToolDefinition]` (populated after first tool call).
Use `fetch_definitions()` to eagerly fetch and cache the list:

```python
defs = await client.tools.fetch_definitions()  # list[ToolDefinition] in OpenAI function format
```

Pass these to any LLM that supports function calling, or let `conversation()` manage them automatically when `"tools": client.tools` is set.

## Models

```python
result = await client.models.list()
for m in result.models:
    print(f"{m['slug']} — ${m['input_price_per_1m']}/1M in, ${m['output_price_per_1m']}/1M out")

# Filter by name
result = await client.models.list(search="claude")
```

`models.list()` returns a `ModelListResult` with `.models` (list) and `.request_id` (str | None).

## Error Handling

All errors are instances of `LLM4AgentsError`:

```python
from llm4agents import LLM4AgentsClient
from llm4agents.errors import LLM4AgentsError

try:
    await client.chat.completions.create(...)
except LLM4AgentsError as err:
    print(err.code, err.status_code, err.request_id, err.message)
```

| `code` | HTTP status | Description |
|---|---|---|
| `auth_error` | 401, 403 | Invalid or missing API key |
| `insufficient_balance` | 402 | Not enough balance to cover the request |
| `rate_limited` | 429 | Too many requests |
| `model_not_found` | 404 | Requested model does not exist in the catalog |
| `model_disabled` | 422 | Model exists but is currently disabled |
| `context_overflow` | — | Prompt + max_tokens exceeds the model's context window |
| `gas_spike` | 409 | Network gas price spiked above safe threshold during transfer |
| `signature_mismatch` | 422 | EIP-712 permit signature could not be verified |
| `invalid_token` | 422 | Unsupported token or chain for gasless transfer |
| `operator_unavailable` | 503 | Gasless relayer is temporarily unavailable |
| `deadline_expired` | 400 | EIP-712 permit deadline passed before submission |
| `tool_not_found` | — | MCP tool name not found in the server's tool list |
| `tool_execution_error` | — | MCP tool returned an error result |
| `tool_loop_limit` | — | Conversation exceeded `max_tool_rounds` without a final answer |
| `network_error` | — | Connection failed (DNS failure, TCP reset, etc.) |
| `timeout` | — | Request exceeded the configured timeout |
| `api_error` | 4xx, 5xx | Any other non-success response |

## Constructor Options

```python
client = LLM4AgentsClient(
    api_key="sk-proxy-...",                         # required
    base_url="https://api.llm4agents.com",          # optional
    mcp_url="https://mcp.llm4agents.com/mcp",       # optional
    timeout=30.0,                                   # optional, seconds, default 30
)
```

## Migration from v1.x

| Before (v1) | After (v2) |
|---|---|
| `result = await tools.call(name, args)` → `str` | `result.text` or `str(result)` |
| `conv.on_tool_result` callback receives `str` | callback now receives `McpToolResult` |
| `models = await client.models.list()` → `list` | `result.models` (access via `.models`) |
| `StreamEvent["tool_end"]["result"]` → `str` | `.result` is now `McpToolResult` |

## License

MIT
