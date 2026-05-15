"""x402 walk-up payment types — public surface of the SDK's x402 module.

Mirrors the proxy's ``src/services/x402Streaming.ts`` constants and the
TypeScript SDK's ``src/x402/types.ts``. See https://x402.org for the wire
protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Literal, Protocol, runtime_checkable

from llm4agents.errors import LLM4AgentsError

# Networks supported by the proxy's x402 wire today.
X402Network = Literal["base", "base-sepolia"]

# CAIP-2 identifiers used internally (mechanisms shape on the wire).
X402_CAIP2_BY_NETWORK: dict[X402Network, str] = {
    "base": "eip155:8453",
    "base-sepolia": "eip155:84532",
}

# USDC contract addresses per network.
USDC_ADDRESS_BY_NETWORK: dict[X402Network, str] = {
    "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "base-sepolia": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
}

# EIP-712 domain ``name()`` differs per chain — verified on-chain by the
# proxy. Base mainnet returns "USD Coin"; Base Sepolia returns "USDC".
# The x402 spec example uses "USDC" everywhere, which silently breaks
# mainnet signatures.
USDC_DOMAIN_NAME_BY_NETWORK: dict[X402Network, str] = {
    "base": "USD Coin",
    "base-sepolia": "USDC",
}

# Chain IDs per supported network.
CHAIN_ID_BY_NETWORK: dict[X402Network, int] = {
    "base": 8453,
    "base-sepolia": 84532,
}


@runtime_checkable
class Signer(Protocol):
    """Minimal signer abstraction.

    Both an ``eth_account.LocalAccount`` (via ``eth_account_to_signer``) and
    any user-supplied adapter (hardware wallet, KMS, WalletConnect, etc.)
    satisfy this protocol. See ``ports & adapters`` rationale in the TS SDK.
    """

    @property
    def address(self) -> str:
        """20-byte EVM address, 0x-prefixed."""

    def sign_typed_data(
        self,
        *,
        domain: dict[str, Any],
        types: dict[str, list[dict[str, str]]],
        primary_type: str,
        message: dict[str, Any],
    ) -> Awaitable[str] | str:
        """Sign EIP-712 typed data.

        The proxy uses EIP-3009 ``TransferWithAuthorization`` for the
        ``exact`` scheme. Return a 65-byte 0x-prefixed signature hex
        string. May be sync or async — the SDK awaits the result if it's
        a coroutine.
        """


@dataclass(frozen=True)
class PaymentConfig:
    """``payment`` constructor option for ``LLM4AgentsClient``.

    When ``mode == 'bearer'`` (default), the client uses
    ``Authorization: Bearer`` with the agent API key. When
    ``mode == 'x402'``, the client probes the proxy on each call to
    ``/v1/chat/completions``, signs an EIP-3009 authorization with the
    provided ``signer``, and sends it via the ``X-PAYMENT`` header.
    """

    mode: Literal["bearer", "x402"] = "bearer"
    signer: Signer | None = None
    network: X402Network | None = None
    pay_to: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "x402" and self.signer is None:
            raise ValueError("PaymentConfig(mode='x402') requires a signer")
        if self.mode == "bearer" and (self.signer is not None or self.network is not None or self.pay_to is not None):
            raise ValueError(
                "PaymentConfig(mode='bearer') cannot carry signer/network/pay_to — use mode='x402'"
            )


@dataclass(frozen=True)
class PaymentRequirements:
    """``PaymentRequirements`` as returned by the proxy in 402 responses.

    Decoded from either the JSON body's ``accepts[]`` or the
    base64-encoded ``PAYMENT-REQUIRED`` response header.
    """

    scheme: str
    network: str
    max_amount_required: str
    asset: str
    pay_to: str
    max_timeout_seconds: int
    resource: str | None = None
    description: str | None = None
    mime_type: str | None = None
    extra: dict[str, Any] | None = None

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> PaymentRequirements:
        """Construct from a wire-format dict.

        The proxy emits camelCase keys (``maxAmountRequired``, ``payTo``,
        ``maxTimeoutSeconds``, ``mimeType``). We map to snake_case.
        """
        return cls(
            scheme=raw["scheme"],
            network=raw["network"],
            max_amount_required=raw["maxAmountRequired"],
            asset=raw["asset"],
            pay_to=raw["payTo"],
            max_timeout_seconds=int(raw.get("maxTimeoutSeconds", 60)),
            resource=raw.get("resource"),
            description=raw.get("description"),
            mime_type=raw.get("mimeType"),
            extra=raw.get("extra"),
        )


@dataclass(frozen=True)
class PaymentPayload:
    """The signed payment payload — the inner content of the ``X-PAYMENT`` header."""

    x402_version: int
    scheme: str
    network: str
    signature: str
    authorization_from: str
    authorization_to: str
    authorization_value: str
    authorization_valid_after: str
    authorization_valid_before: str
    authorization_nonce: str

    def to_wire(self) -> dict[str, Any]:
        """Serialize to the on-the-wire shape (camelCase, nested payload)."""
        return {
            "x402Version": self.x402_version,
            "scheme": self.scheme,
            "network": self.network,
            "payload": {
                "signature": self.signature,
                "authorization": {
                    "from": self.authorization_from,
                    "to": self.authorization_to,
                    "value": self.authorization_value,
                    "validAfter": self.authorization_valid_after,
                    "validBefore": self.authorization_valid_before,
                    "nonce": self.authorization_nonce,
                },
            },
        }


@dataclass(frozen=True)
class X402Receipt:
    """Receipt returned to the caller after a successful x402 settlement.

    Surfaced via the trailing ``event: x402-receipt`` SSE chunk in
    streaming mode (yielded as a ``{"type": "x402_receipt", ...}`` event
    on ``Conversation.stream()``) and via the ``on_x402_receipt``
    callback on ``ChatCompletions.create()``.
    """

    transaction: str
    network: str
    amount: str
    payer: str


class X402PaymentRequiredError(LLM4AgentsError):
    """Thrown on HTTP 402 responses carrying x402 ``paymentRequirements``.

    Distinct from ``insufficient_balance`` (which means a Bearer agent's
    pre-deposited balance is too low). Carries the typed requirements
    list so callers can re-sign with a different amount or recipient.
    """

    def __init__(
        self,
        message: str,
        payment_requirements: list[PaymentRequirements],
        x402_version: int,
        status_code: int | None,
        request_id: str | None,
    ) -> None:
        super().__init__(message, "x402_payment_required", status_code, request_id)
        self.payment_requirements = payment_requirements
        self.x402_version = x402_version
