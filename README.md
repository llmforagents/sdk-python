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

## x402 Walk-up Payment

The proxy supports the [x402 protocol](https://x402.org) for per-request stablecoin
payments on `POST /v1/chat/completions`. Instead of pre-funding an agent account,
the client signs an EIP-3009 `TransferWithAuthorization` for USDC on Base /
Base-Sepolia, attaches it as an `X-PAYMENT` header, and the proxy settles
on-chain after the response is delivered.

Two modes are mutually exclusive — pick one at construction time:

| Mode | Set via | Required | Use when |
|---|---|---|---|
| **Bearer** (default) | omit `payment` or `PaymentConfig(mode="bearer")` | `api_key` | You have an agent and a pre-funded balance |
| **x402 walk-up** | `payment=PaymentConfig(mode="x402", signer=...)` | `signer` (eth_account or custom) | You want one-shot calls billed per-request from a wallet, no agent registration |

### Bearer vs x402 — at a glance

```python
from llm4agents import LLM4AgentsClient, PaymentConfig, eth_account_to_signer
from eth_account import Account

# Bearer (existing) — pre-funded agent
bearer = LLM4AgentsClient(api_key="sk-proxy-...")

# x402 walk-up — pay per call from a wallet
account = Account.from_key("0xYOUR_PRIVATE_KEY")
x402 = LLM4AgentsClient(
    api_key="",  # ignored in x402 mode
    payment=PaymentConfig(
        mode="x402",
        signer=eth_account_to_signer(account),
        network="base-sepolia",        # or "base" for mainnet
    ),
)

# Same API surface — the SDK probes the proxy for a 402, signs an
# EIP-3009 authorization, and retries with X-PAYMENT automatically.
res = await x402.chat.completions.create({
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}],
})
```

### Custom signers (no eth_account dependency)

`eth_account` is already a hard dependency of the SDK (used by gasless transfers),
so `eth_account_to_signer` is the default path. If you want to plug in a hardware
wallet, KMS-backed key, or WalletConnect, implement the `Signer` Protocol
directly — the SDK depends on the Protocol, not on `eth_account`:

```python
from llm4agents import Signer

class HsmSigner:
    address = "0xYourAddress..."

    async def sign_typed_data(self, *, domain, types, primary_type, message):
        # Defer to your HSM / KMS / hardware wallet. Return a 0x-prefixed
        # 65-byte signature.
        return "0x..."

client = LLM4AgentsClient(
    api_key="",
    payment=PaymentConfig(mode="x402", signer=HsmSigner(), network="base"),
)
```

`sign_typed_data` may be sync or async — the SDK awaits the result if it's
a coroutine.

### Streaming receipts

x402-mode streaming responses end with a trailing SSE event after `[DONE]`
containing the on-chain settlement receipt. The `Conversation.stream()` helper
surfaces this as a typed event yielded BEFORE the matching `done` event:

```python
conv = x402.chat.conversation({"model": "openai/gpt-4o-mini"})
async for ev in conv.stream("Tell me a joke"):
    if ev["type"] == "text":
        print(ev["content"], end="", flush=True)
    elif ev["type"] == "x402_receipt":
        print(f"\nsettled: {ev['transaction']} on {ev['network']}")
    elif ev["type"] == "done":
        print(f"\n{ev['response']['usage']}")
```

For the lower-level `chat.completions.create()` API, pass `on_x402_receipt`:

```python
def handle_receipt(receipt):
    print(f"settled {receipt.amount} on {receipt.network}: {receipt.transaction}")

stream = await x402.chat.completions.create(
    {"model": "openai/gpt-4o-mini", "messages": [...], "stream": True},
    on_x402_receipt=handle_receipt,
)
async for chunk in stream:
    ...
```

### Lower-level helpers — `client.x402`

For advanced use cases (signing without sending, inspecting the 402 response
shape, batch signing), the `client.x402` namespace exposes the building blocks:

```python
# Probe the proxy and get the typed PaymentRequirements
requirements = await x402.x402.probe()
print(requirements.max_amount_required, requirements.network)

# Probe + sign in one call — returns SignedPayment(payment_payload, encoded_header, requirements)
signed = await x402.x402.sign()
#   signed.encoded_header     → base64-encoded X-PAYMENT value
#   signed.payment_payload    → the parsed PaymentPayload (typed)
#   signed.requirements       → the proxy-advertised requirements the signature is bound to

# Sign against caller-supplied requirements (no HTTP) — useful for testing
signed2 = await x402.x402.sign_from_requirements(requirements)
```

### Error handling

When the proxy rejects payment (signature invalid, nonce reused, etc.) the SDK
raises `X402PaymentRequiredError` carrying the typed requirements so the caller
can re-sign with a different amount or network:

```python
from llm4agents import X402PaymentRequiredError

try:
    await x402.chat.completions.create({...})
except X402PaymentRequiredError as err:
    print("Payment rejected. accepted offers:", err.payment_requirements)
    print("x402 version:", err.x402_version)
```

> **Networks:** `"base"` (mainnet, USDC `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`)
> and `"base-sepolia"` (testnet, USDC `0x036CbD53842c5426634e7929541eC2318f3dCF7e`)
> are currently supported. The USDC EIP-712 domain name differs between them
> (`USD Coin` vs `USDC`); `eth_account_to_signer` handles this automatically.

> **Endpoints accepting x402** (signed per-call USDC):
> - `POST /v1/chat/completions` — chat with any model (per-token signed upper bound)
> - `POST /v1/scrape/{markdown,fetch_html,links,screenshot,pdf,extract}` — one-shot scraping
> - `POST /v1/search/{google,news,maps,batch}` — Google search (Serper)
> - `POST /v1/image/{generate,edit,analyze}` — image generation / edit / vision
>
> Per-call x402 prices are seeded ~10% below x402engine.app reference rates
> (e.g. scrape markdown ~$0.0045, screenshot ~$0.009, image gen ~$0.0135-$0.045).
> Prices are admin-editable from the operator panel without redeploy.
>
> Browser sessions (`session_*`) and other endpoints (`/v1/embeddings`,
> `/api/v1/wallets/*`, etc.) stay **Bearer-only** — sessions are
> pre-deposit by design. Using x402 mode on a non-allowed path raises
> `LLM4AgentsError` with code `x402_payment_required` and a clear message.

### REST scrape / search / image with x402

The same `payment=PaymentConfig(mode="x402", signer=..., network=...)`
client config that works for chat completions also works for the MCP
REST surface:

```python
import httpx
from eth_account import Account
from llm4agents import LLM4AgentsClient, PaymentConfig, eth_account_to_signer

account = Account.from_key("0xYOUR_KEY")
client = LLM4AgentsClient(
    api_key="",
    payment=PaymentConfig(
        mode="x402",
        signer=eth_account_to_signer(account),
        network="base-sepolia",
    ),
)

# Probe + sign once, then hit the REST endpoint directly with the X-PAYMENT
signed = await client.x402.sign()
async with httpx.AsyncClient() as http:
    res = await http.post(
        "https://api.llm4agents.com/v1/scrape/markdown",
        headers={"x-payment": signed.encoded_header},
        json={"url": "https://example.com"},
    )
    print(res.json())
```

The MCP tools accessor (`client.tools.scraper.markdown(...)`) currently
uses Bearer auth via the MCP transport; the REST surface above is the
path for walk-up.

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

## Embeddings

```python
res = await client.embeddings.create(
    model="openai/text-embedding-3-large",
    input="How many vectors fit in a haystack?",
)
print(len(res.data[0].embedding))  # → e.g. 3072
print(res.usage.prompt_tokens, res.model)

# Batch input
batch = await client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=["first", "second", "third"],
)
for item in batch.data:
    print(item.index, item.embedding)
```

`embeddings.create()` accepts `model` (slug), `input` (string or list of strings, max 2048 entries), and the optional `encoding_format`, `dimensions`, and `user` keyword args. Returns an `EmbeddingsResponse` with `.data: list[EmbeddingItem]`, `.model`, and `.usage` (`prompt_tokens`, `total_tokens`). Embeddings have no completion tokens, so billing is input-only.

> **Catalog:** Embedding models do not appear in OpenRouter's public catalog endpoint, so the proxy maintains them by hand. New embedding models can be added through the admin panel — see `model_type='embedding'` rows.

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
    api_key="sk-proxy-...",                         # required in Bearer mode; "" in x402 mode
    base_url="https://api.llm4agents.com",          # optional
    mcp_url="https://mcp.llm4agents.com/mcp",       # optional
    timeout=30.0,                                   # optional, seconds, default 30
    payment=PaymentConfig(mode="bearer"),           # optional, default; or PaymentConfig(mode="x402", signer=..., network=...)
)
```

## What's New in v2.5

- **x402 walk-up payment mode** — pay per-request from a wallet on `/v1/chat/completions` without
  registering an agent. Pass `payment=PaymentConfig(mode="x402", signer=..., network=...)` to the
  client constructor. Supports both `eth_account.LocalAccount` (via `eth_account_to_signer`) and
  any custom `Signer` Protocol implementation (HSM, KMS, hardware wallets, WalletConnect) thanks
  to the Ports & Adapters design — `sign_typed_data` may be sync or async.
- Streaming responses emit a typed `x402_receipt` event after `[DONE]`, surfacing the on-chain
  settlement receipt (`transaction`, `network`, `amount`, `payer`) to `conv.stream()` consumers
  and the `on_x402_receipt` callback on `chat.completions.create()`.
- New `client.x402` namespace — `probe()`, `sign(recipient=...)`, and
  `sign_from_requirements(req, recipient=...)` helpers for low-level integrations.
- New top-level exports: `PaymentConfig`, `PaymentPayload`, `PaymentRequirements`, `Signer`,
  `SignedPayment`, `X402Namespace`, `X402Network`, `X402PaymentRequiredError`, `X402Receipt`,
  `eth_account_to_signer`, `build_transfer_with_authorization_typed_data`, `generate_nonce`,
  `network_to_caip2`, `sign_from_requirements`, `encode_payment_header`,
  `decode_payment_required_header`, `pick_supported_requirements`, `USDC_ADDRESS_BY_NETWORK`,
  `USDC_DOMAIN_NAME_BY_NETWORK`, `X402_CAIP2_BY_NETWORK`, `CHAIN_ID_BY_NETWORK`,
  `TRANSFER_WITH_AUTHORIZATION_TYPES`, `DEFAULT_VALID_FOR_SECONDS`.
- **x402 allowlist extended** to the MCP REST surface — clients in x402
  mode can now hit `/v1/scrape/*`, `/v1/search/*`, and `/v1/image/*` in
  addition to chat. Prices are admin-editable in cents from the
  operator panel (parallel `value` for balance / `x402_value` for
  walk-up per tool).

## What's New in v2.4

- **`client.embeddings.create(model=..., input=...)`** — OpenAI-compatible embeddings against `POST /v1/embeddings`. Pass a string or a list of up to 2048 strings; receive an `EmbeddingsResponse` with `data: list[EmbeddingItem]`, `model`, and `usage` (prompt_tokens, total_tokens). Embeddings are billed input-only — there are no completion tokens.
- New types exported from `llm4agents`: `Embeddings`, `EmbeddingItem`, `EmbeddingsResponse`, `EmbeddingsUsage`. The embedding-model catalog is curated by hand on the server because OpenRouter omits embedding models from its public catalog endpoint; admins maintain the list via the proxy admin panel.

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
