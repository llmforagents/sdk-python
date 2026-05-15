"""``client.x402`` namespace — low-level helpers for x402 walk-up payments.

``sign(...)`` probes the proxy for live ``PaymentRequirements`` and signs
over them. ``sign_from_requirements(...)`` skips the probe (faster, but
the caller must pass valid requirements). ``probe()`` exposes the probe
step on its own so users can introspect the proxy's current pricing
before deciding whether to sign.

Only available when the client is constructed with
``payment.mode == 'x402'``. Constructing without ``payment`` (Bearer
mode) raises ``LLM4AgentsError`` on first use.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from llm4agents.errors import LLM4AgentsError
from llm4agents.x402.payment import (
    SignedPayment,
    decode_payment_required_header,
    pick_supported_requirements,
    sign_from_requirements,
)
from llm4agents.x402.types import (
    PaymentConfig,
    PaymentRequirements,
    Signer,
    X402Network,
)


class X402Namespace:
    """``client.x402`` — only useful when ``payment.mode == 'x402'``."""

    def __init__(self, payment: PaymentConfig, base_url: str, timeout: float) -> None:
        self._payment = payment
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def _signer(self) -> Signer:
        if self._payment.mode != "x402" or self._payment.signer is None:
            raise LLM4AgentsError(
                "client.x402 is only available when LLM4AgentsClient is "
                "constructed with payment=PaymentConfig(mode='x402', ...)",
                "x402_payment_required",
                None,
                None,
            )
        return self._payment.signer

    @property
    def _network(self) -> X402Network:
        if self._payment.mode != "x402":
            raise LLM4AgentsError(
                "client.x402 is only available when LLM4AgentsClient is "
                "constructed with payment=PaymentConfig(mode='x402', ...)",
                "x402_payment_required",
                None,
                None,
            )
        return self._payment.network or "base"

    async def probe(self) -> PaymentRequirements:
        """Probe the proxy for current ``PaymentRequirements`` without signing.

        Issues an unauthenticated POST to ``/v1/chat/completions`` with a
        minimal probe body and reads the 402 response.
        """
        # Force-evaluate the property so misuse fails fast before HTTP.
        _ = self._signer

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            res = await client.post(
                f"{self._base_url}/v1/chat/completions",
                headers={"content-type": "application/json"},
                content=json.dumps({"messages": [{"role": "user", "content": "probe"}]}),
            )
        if res.status_code != 402:
            text = res.text[:200] if res.text else ""
            raise LLM4AgentsError(
                f"Expected HTTP 402 from probe, got {res.status_code}: {text}",
                "api_error",
                res.status_code,
                res.headers.get("x-request-id"),
            )

        header_value = res.headers.get("payment-required")
        if header_value is not None:
            _version, accepts = decode_payment_required_header(header_value)
            return pick_supported_requirements(accepts)

        try:
            body = res.json()
        except ValueError:
            raise LLM4AgentsError(
                "Probe response had no PAYMENT-REQUIRED header and a "
                "non-JSON body",
                "api_error",
                res.status_code,
                res.headers.get("x-request-id"),
            ) from None
        accepts_raw: Any = body.get("accepts") if isinstance(body, dict) else None
        if not isinstance(accepts_raw, list):
            raise LLM4AgentsError(
                "Probe response had no PAYMENT-REQUIRED header and no "
                "parseable accepts[] in body",
                "api_error",
                res.status_code,
                res.headers.get("x-request-id"),
            )
        return pick_supported_requirements(accepts_raw)

    async def sign(self, *, recipient: str | None = None) -> SignedPayment:
        """Probe + sign. Returns the encoded ``X-PAYMENT`` header value and
        the fully-formed payload object."""
        requirements = await self.probe()
        return await self.sign_from_requirements(requirements, recipient=recipient)

    async def sign_from_requirements(
        self,
        requirements: PaymentRequirements,
        *,
        recipient: str | None = None,
    ) -> SignedPayment:
        """Sign against caller-supplied requirements. No HTTP traffic.

        Useful for testing, batch signing, or when the caller already
        fetched the requirements via ``probe()`` and wants to reuse
        them across calls.
        """
        return await sign_from_requirements(
            signer=self._signer,
            network=self._network,
            requirements=requirements,
            recipient_override=recipient,
        )
