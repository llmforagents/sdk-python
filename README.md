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
from llm4agents import FinalUsage

def on_final_usage(usage: FinalUsage) -> None:
    print(f"\n[usage] prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")

stream = await client.chat.completions.create(
    {
        "model": "anthropic/claude-sonnet-4",
        "messages": [{"role": "user", "content": "Count to 10"}],
        "stream": True,
    },
    on_final_usage=on_final_usage,   # fires once after the stream ends
)
async for chunk in stream:
    print(chunk.choices[0]["delta"].get("content", ""), end="", flush=True)

# With extended thinking
response = await client.chat.completions.create({
    "model": "anthropic/claude-sonnet-4",
    "messages": [{"role": "user", "content": "Solve step by step: 47 * 83"}],
    "reasoning": True,
    "include_reasoning": True,
})

# Vision (multimodal) input
analysis = await client.chat.completions.create({
    "model": "openai/gpt-4o",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        ],
    }],
})

# Model fallbacks — try primary, fall back to secondary if it fails
response = await client.chat.completions.create({
    "models": ["anthropic/claude-sonnet-4", "openai/gpt-4o"],
    "messages": [{"role": "user", "content": "Hello"}],
})
# X-Model-Used response header indicates which model actually responded

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
from llm4agents import LLM4AgentsError, ResponseMeta

def on_round_meta(meta: ResponseMeta) -> None:
    cents = meta.cost_usd_cents or 0
    bal = meta.balance_remaining_cents or 0
    print(f"Round cost: ${cents / 100:.4f}, balance: ${bal / 100:.2f}")

conv = client.chat.conversation({
    "model": "anthropic/claude-sonnet-4",
    "system": "You are a research assistant",
    "tools": client.tools,
    "history": [],                # optional: rehydrate from persisted messages
    "on_tool_call": lambda name, args: print(f"Calling {name}...") or True,
    "on_tool_result": lambda name, result: print(f"{name}: {result.text[:80]}"),
    "on_round_meta": on_round_meta,
    "on_tools_ignored": lambda model: print(f"WARN: {model} ignored tools"),
    "enable_prompt_tool_fallback": True,   # retry round 0 with tools in prompt
    "max_tool_rounds": 5,         # default 10
})

# Single turn
answer = await conv.say("Search for Bitcoin news and summarize the top 3")
print(answer.content)
for call in answer.tool_calls:
    print(f"  {call['name']} → {call['result'].text[:60]}")

# Streaming conversation
async for event in conv.stream("Now find the current price"):
    if event["type"] == "text":
        print(event["content"], end="", flush=True)
    elif event["type"] == "reasoning":
        print(f"<think>{event['content']}</think>", end="")
    elif event["type"] == "meta":
        print(f"\n[meta] request_id={event['meta'].request_id}")
    elif event["type"] == "tool_call":
        print(f"\n[tool] {event['name']}({event['args']})")
    elif event["type"] == "tool_result":
        print(f"[done] {event['result'].text[:60]}")
    elif event["type"] == "done":
        print(f"\nUsage: {event['response']['usage']}")

# History management
messages = conv.messages    # list[ChatMessage] — JSON-serializable
conv.clear()                # reset to empty, keeps system prompt and options
branch = conv.fork()        # copy history + all callbacks into a new Conversation
```

`on_tool_call` receives `(name: str, args: dict)` and should return `True` to proceed or `False` to cancel the tool call.

`on_round_meta` fires after each round with a `ResponseMeta` containing `cost_usd_cents`, `balance_remaining_cents`, `tokens_input`, `tokens_output`, `tokens_reasoning`, `model_used`, and `request_id` parsed from response headers.

`on_tools_ignored(model)` fires once if you pass `tools` but the model returns no tool calls on the first round — useful for detecting models without native function calling.

`enable_prompt_tool_fallback=True` adds an automatic recovery path for those models. When round 0 ignores tools, the SDK retries the round once with the tool definitions injected into the system prompt and asks the model to emit `<tool_call>{"name":"...","arguments":{...}}</tool_call>` blocks. Parsed blocks are executed exactly like native tool calls, so the rest of the loop is unchanged. In `stream()` you'll see a `{"type": "fallback", "reason": "tools_ignored", "model": ...}` event before the fallback round runs (the fallback round itself is non-streamed). If the fallback also returns no blocks, its plain-text reply is returned as the answer.

`ConversationResponse.usage` includes `prompt_tokens`, `completion_tokens`, `total_tokens`, and (when the upstream model exposes them) `reasoning_tokens` accumulated across every round, including any fallback round.

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
        print(part.mimeType, len(part.data))    # base64-encoded image
    elif part.type == "resource":
        print(part.uri, part.text)
```

The transport auto-normalizes raw MCP responses: snake_case `mime_type` is aliased to `mimeType`, `imageBase64`/`pngBase64` keys are mapped to `data`, MIME types are sniffed from base64 magic bytes when missing, and JSON-wrapped image/PDF payloads embedded inside text blocks (e.g. `{"imageBase64": "...", "mimeType": "image/png"}`) are promoted to typed `McpImageContent` / `McpResourceContent` automatically.

## Agents

```python
# Register a new agent — call before you have an API key
client = LLM4AgentsClient(api_key="")   # empty key is fine for registration
reg = await client.agents.register("My Agent")
# The returned api_key is shown only once — save it immediately
print(reg.api_key)        # sk-proxy-...
print(reg.uuid)
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

`type` filter accepts `"deposit"`, `"usage"`, `"refund"`, or `"gas_sponsored"`.

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
print(f"Fee: {quote.fee_formatted}")
print(f"Forwarder: {quote.forwarder_address}")

result = await client.transfer.submit(quote, "0xPrivateKey...")
print(result.tx_hash)
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
    print(f"{m['slug']} — ${m['inputPricePer1M']}/1M in, ${m['outputPricePer1M']}/1M out")
    if m.get("feePct") is not None:
        print(f"  platform fee: {m['feePct']}%")

# Filter by name
result = await client.models.list(search="claude")
```

`models.list()` returns a `ModelListResult` with `.models` (list of dicts), `.fee_pct` (int | None — the agent's platform fee percentage, applied to every billed call), and `.request_id` (str | None). Each model dict contains `slug`, `displayName`, `provider`, `inputPricePer1M`, `outputPricePer1M`, `contextWindow`, `lastSyncedAt`, and an optional `feePct` (platform fee percentage).

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

## What's New in v2.3

This release brings the Python SDK to feature parity with the TypeScript SDK at v2.3.1. Seven fixes that an external playground audit flagged as real-world breakages:

- **Prompt-mode tool fallback** (`enable_prompt_tool_fallback=True`) — when a model ignores native `tools` on round 0, the SDK retries the round with the tool definitions injected into the system prompt and parses `<tool_call>{"name":"...","arguments":{...}}</tool_call>` blocks from the response. Recovers tool use on models without native function calling. `stream()` emits a `{"type": "fallback", "reason": "tools_ignored", "model": ...}` event before the fallback round.
- **MCP `Accept` header** — MCP `rpc` requests now send `Accept: application/json, text/event-stream`. Streamable HTTP MCP servers were rejecting every tool call with HTTP 406 without it.
- **Models endpoint trailing slash** — `client.models.list()` now hits `/api/v1/models` (no trailing slash), matching the deployed API.
- **`reasoning_tokens` propagation** — `ConversationResponse.usage["reasoning_tokens"]` now accumulates across all rounds, and `ResponseMeta.tokens_reasoning` exposes the per-round value parsed from the `x-tokens-reasoning` header.
- **`fee_pct` in `ModelListResult`** — the platform fee percentage returned by the API is now surfaced as `result.fee_pct: int | None`.
- **`on_final_usage` callback in streaming completions** — `chat.completions.create(params, on_final_usage=cb)` fires `cb(FinalUsage(prompt_tokens, completion_tokens, total_tokens, reasoning_tokens))` once after the SSE stream ends, using the last `usage` chunk emitted by providers that send `stream_options: {include_usage: true}`.
- **Assistant `content: null` normalization** — when a model returns `content: null` for a tool-only message (OpenAI / Gemini convention), the SDK now normalizes it to `""` before pushing to history. Strict backends were rejecting follow-up requests because `messages[*].content` was null.

## What's New in v2.1

- **`Conversation.stream()`** — async generator yielding typed events: `text`, `reasoning`, `tool_call`, `tool_result`, `meta`, `done`. Each round emits a `meta` event with `ResponseMeta` parsed from response headers.
- **`on_round_meta`** — callback fired per round with `ResponseMeta` (cost, balance, tokens, model used, request id).
- **`on_tools_ignored(model)`** — callback fired when a model returns no tool calls on the first round despite being given tools — useful for detecting models without native function-calling support.
- **`agents.register(name)`** — register a new agent and receive its `api_key` (only shown once).
- **`forwarder_address`** in `QuoteResult` — the EIP-2771 forwarder contract used for the gasless transfer.
- **Auto-normalization in MCP transport** — JSON-wrapped image/PDF payloads embedded inside text blocks (`{"imageBase64": "..."}`, `{"pngBase64": "..."}`, `{"pdfBase64": "..."}`) are auto-promoted to typed `McpImageContent` / `McpResourceContent`. Snake_case `mime_type` is aliased to `mimeType` and MIME types are sniffed from base64 magic bytes when missing.
- **Top-level exports** — `from llm4agents import AgentRegistration, ResponseMeta, Conversation, McpToolResult, ...` (all public types now re-exported from the package root).
- **`Transaction.type`** filter accepts `"gas_sponsored"` in addition to `"deposit"`, `"usage"`, `"refund"`.
- **`ModelInfo.feePct`** — platform fee percentage now exposed in the model catalog.

## Migration from v1.x

| Before (v1) | After (v2) |
|---|---|
| `result = await tools.call(name, args)` → `str` | `result.text` or `str(result)` |
| `conv.on_tool_result` callback receives `str` | callback now receives `McpToolResult` |
| `models = await client.models.list()` → `list` | `result.models` (access via `.models`) |
| `StreamEvent["tool_end"]["result"]` → `str` | `.result` is now `McpToolResult` |

## License

MIT
