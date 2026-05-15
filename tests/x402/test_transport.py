"""Tests for HttpTransport's x402 probe-and-sign integration."""
from __future__ import annotations

import base64
import json

import httpx
import pytest
import respx
from eth_account import Account

from llm4agents.errors import LLM4AgentsError
from llm4agents.transport.http import HttpTransport
from llm4agents.x402.payment import encode_payment_header
from llm4agents.x402.signer import eth_account_to_signer
from llm4agents.x402.types import (
    PaymentConfig,
    PaymentPayload,
    X402PaymentRequiredError,
)

TEST_KEY = "0x" + "1" * 64


def _requirements_body() -> dict:
    return {
        "x402Version": 2,
        "accepts": [
            {
                "scheme": "exact",
                "network": "eip155:84532",
                "maxAmountRequired": "10000",
                "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                "payTo": "0x0000000000000000000000000000000000000033",
                "maxTimeoutSeconds": 60,
                "extra": {"name": "USDC", "version": "2"},
            }
        ],
    }


@respx.mock
async def test_bearer_mode_regression_sends_authorization_header() -> None:
    route = respx.post("https://api.test/api/v1/balance").mock(
        return_value=httpx.Response(200, json={"ok": True}),
    )
    http = HttpTransport("https://api.test", "sk-test", 5.0)
    await http.post("/api/v1/balance", {})
    assert route.called
    req = route.calls[0].request
    assert req.headers.get("authorization") == "Bearer sk-test"
    assert "x-payment" not in {k.lower() for k in req.headers.keys()}


@respx.mock
async def test_x402_mode_probes_then_signs_and_retries_with_x_payment() -> None:
    signer = eth_account_to_signer(Account.from_key(TEST_KEY))
    # 1st: probe → 402 with paymentRequirements in body
    # 2nd: signed call → 200
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(402, json=_requirements_body()),
            httpx.Response(200, json={"id": "gen-test", "choices": []}),
        ]
    )

    http = HttpTransport(
        "https://api.test",
        "",
        5.0,
        payment=PaymentConfig(mode="x402", signer=signer, network="base-sepolia"),
    )
    res = await http.post(
        "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert res == {"id": "gen-test", "choices": []}

    # First call (probe) — no auth, no x-payment
    probe_req = respx.routes[0].calls[0].request
    headers_lower = {k.lower(): v for k, v in probe_req.headers.items()}
    assert "authorization" not in headers_lower
    assert "x-payment" not in headers_lower

    # Second call (signed) — x-payment present, no authorization
    signed_req = respx.routes[0].calls[1].request
    headers_lower = {k.lower(): v for k, v in signed_req.headers.items()}
    assert "authorization" not in headers_lower
    assert "x-payment" in headers_lower
    decoded = json.loads(base64.b64decode(headers_lower["x-payment"]).decode("utf-8"))
    assert decoded["scheme"] == "exact"
    assert (
        decoded["payload"]["authorization"]["from"].lower()
        == signer.address.lower()
    )


@respx.mock
async def test_x402_mode_produces_unique_nonces_across_calls() -> None:
    signer = eth_account_to_signer(Account.from_key(TEST_KEY))
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(402, json=_requirements_body()),
            httpx.Response(200, json={"id": "a"}),
            httpx.Response(402, json=_requirements_body()),
            httpx.Response(200, json={"id": "b"}),
        ]
    )
    http = HttpTransport(
        "https://api.test",
        "",
        5.0,
        payment=PaymentConfig(mode="x402", signer=signer, network="base-sepolia"),
    )
    await http.post("/v1/chat/completions", {"messages": []})
    await http.post("/v1/chat/completions", {"messages": []})

    calls = respx.routes[0].calls
    sig1 = json.loads(
        base64.b64decode(calls[1].request.headers["x-payment"]).decode("utf-8")
    )
    sig2 = json.loads(
        base64.b64decode(calls[3].request.headers["x-payment"]).decode("utf-8")
    )
    assert (
        sig1["payload"]["authorization"]["nonce"]
        != sig2["payload"]["authorization"]["nonce"]
    )


@respx.mock
async def test_x402_mode_raises_typed_error_on_signed_call_402() -> None:
    """When the facilitator rejects the signature (e.g., expired nonce),
    the SDK must surface ``X402PaymentRequiredError`` — not a generic
    ``insufficient_balance``."""
    signer = eth_account_to_signer(Account.from_key(TEST_KEY))
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(402, json=_requirements_body()),
            httpx.Response(402, json=_requirements_body()),  # signed call also 402
        ]
    )
    http = HttpTransport(
        "https://api.test",
        "",
        5.0,
        payment=PaymentConfig(mode="x402", signer=signer, network="base-sepolia"),
    )
    with pytest.raises(X402PaymentRequiredError):
        await http.post("/v1/chat/completions", {})


@respx.mock
async def test_bearer_mode_raises_typed_error_on_x402_shaped_402() -> None:
    """Even Bearer-mode clients should surface a typed error when the
    server emits an x402-shaped 402 (e.g., during a proxy migration)."""
    accepts_blob = base64.b64encode(
        json.dumps(_requirements_body()).encode("utf-8")
    ).decode("ascii")
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            402,
            json={"error": "payment required"},
            headers={"payment-required": accepts_blob},
        )
    )
    http = HttpTransport("https://api.test", "sk-test", 5.0)
    with pytest.raises(X402PaymentRequiredError):
        await http.post("/v1/chat/completions", {})


async def test_x402_mode_blocks_non_allowed_paths() -> None:
    signer = eth_account_to_signer(Account.from_key(TEST_KEY))
    http = HttpTransport(
        "https://api.test",
        "",
        5.0,
        payment=PaymentConfig(mode="x402", signer=signer, network="base-sepolia"),
    )
    with pytest.raises(LLM4AgentsError, match="x402 mode is only available"):
        await http.post("/api/v1/wallets/generate", {})
    with pytest.raises(LLM4AgentsError, match="x402 mode is only available"):
        await http.post("/v1/embeddings", {})


@respx.mock
@pytest.mark.parametrize("path", [
    "/v1/scrape/markdown",
    "/v1/scrape/fetch_html",
    "/v1/search/google",
    "/v1/image/generate",
])
async def test_x402_mode_allows_mcp_rest_paths(path: str) -> None:
    """Verify the allowlist accepts every REST surface from the proxy
    P3 wire-up: /v1/scrape/*, /v1/search/*, /v1/image/*. Each probes,
    receives 402 + paymentRequirements, signs, retries, gets 200."""
    signer = eth_account_to_signer(Account.from_key(TEST_KEY))
    respx.post(f"https://api.test{path}").mock(
        side_effect=[
            httpx.Response(402, json=_requirements_body()),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    http = HttpTransport(
        "https://api.test",
        "",
        5.0,
        payment=PaymentConfig(mode="x402", signer=signer, network="base-sepolia"),
    )
    result = await http.post(path, {})
    assert result == {"ok": True}
