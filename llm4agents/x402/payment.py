"""Payment-payload assembly + header (de)serialization for x402.

Two flows:

  1. ``sign_from_requirements(...)`` — caller already has
     ``PaymentRequirements`` (e.g. fetched via ``client.x402.probe()``
     or parsed from a 402 response). Signs and returns the encoded
     ``X-PAYMENT`` header value + the parsed payload.
  2. ``HttpTransport.resolve_auth_headers(...)`` (in ``transport/http.py``)
     — probes the proxy itself and chains into ``sign_from_requirements``.

Both return ``SignedPayment`` so users who want full control (the
low-level helper exposed as ``client.x402.sign``) can introspect the
signed payload.
"""
from __future__ import annotations

import base64
import inspect
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from llm4agents.x402.signer import (
    build_transfer_with_authorization_typed_data,
    generate_nonce,
)
from llm4agents.x402.types import (
    PaymentPayload,
    PaymentRequirements,
    Signer,
    X402Network,
)

# 5 minutes — generous for slow clients, well within facilitator quotas.
DEFAULT_VALID_FOR_SECONDS = 5 * 60

_EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class SignedPayment:
    """Result of a signed x402 authorization.

    ``encoded_header`` is what the SDK attaches as the ``X-PAYMENT`` HTTP
    header. ``payment_payload`` is the parsed structure for logging /
    debugging. ``requirements`` is the ``PaymentRequirements`` the
    signature is bound to (used by callers that want to display
    "you're paying X for Y" to the user).
    """

    payment_payload: PaymentPayload
    encoded_header: str
    requirements: PaymentRequirements


async def sign_from_requirements(
    *,
    signer: Signer,
    network: X402Network,
    requirements: PaymentRequirements,
    recipient_override: str | None = None,
) -> SignedPayment:
    """Sign a payment authorization against caller-supplied requirements.

    No HTTP traffic. Returns the full payload + the encoded X-PAYMENT
    header value (base64-of-JSON).
    """
    recipient = recipient_override or _assert_hex_address(requirements.pay_to)
    now_sec = int(time.time())
    valid_after = "0"
    valid_before = str(now_sec + DEFAULT_VALID_FOR_SECONDS)
    nonce = generate_nonce()

    typed_data = build_transfer_with_authorization_typed_data(
        signer=signer,
        network=network,
        to=recipient,
        value=requirements.max_amount_required,
        valid_after=valid_after,
        valid_before=valid_before,
        nonce=nonce,
    )

    signature = signer.sign_typed_data(
        domain=typed_data["domain"],
        types=typed_data["types"],
        primary_type=typed_data["primaryType"],
        message=typed_data["message"],
    )
    if inspect.isawaitable(signature):
        signature = await signature
    if not isinstance(signature, str):
        raise TypeError(
            f"signer.sign_typed_data must return a str (got {type(signature).__name__})"
        )

    payment_payload = PaymentPayload(
        x402_version=1,
        scheme=requirements.scheme,
        network=requirements.network,
        signature=signature,
        authorization_from=signer.address,
        authorization_to=recipient,
        authorization_value=requirements.max_amount_required,
        authorization_valid_after=valid_after,
        authorization_valid_before=valid_before,
        authorization_nonce=nonce,
    )

    return SignedPayment(
        payment_payload=payment_payload,
        encoded_header=encode_payment_header(payment_payload),
        requirements=requirements,
    )


def encode_payment_header(payload: PaymentPayload) -> str:
    """Base64-encode a ``PaymentPayload`` for the ``X-PAYMENT`` header."""
    return base64.b64encode(json.dumps(payload.to_wire()).encode("utf-8")).decode("ascii")


def decode_payment_required_header(value: str) -> tuple[int, list[PaymentRequirements]]:
    """Decode a base64 ``PAYMENT-REQUIRED`` response header.

    Returns ``(x402Version, accepts[])``. The proxy emits this header on
    402 responses alongside the JSON body.
    """
    raw = base64.b64decode(value).decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed.get("x402Version"), int):
        raise ValueError("PAYMENT-REQUIRED header: missing or non-numeric x402Version")
    accepts = parsed.get("accepts")
    if not isinstance(accepts, list):
        raise ValueError("PAYMENT-REQUIRED header: accepts is not an array")
    return parsed["x402Version"], [PaymentRequirements.from_wire(r) for r in accepts]


def pick_supported_requirements(
    accepts: list[PaymentRequirements] | list[dict[str, Any]],
) -> PaymentRequirements:
    """Pick the first scheme this SDK supports from a list of requirements.

    Accepts either pre-parsed ``PaymentRequirements`` (from
    ``decode_payment_required_header``) or raw wire dicts (from a 402
    body's ``accepts[]``). Today only ``exact`` is supported; ``upto``
    (Permit2) is out of scope per the SDK plan.
    """
    normalized: list[PaymentRequirements] = []
    for entry in accepts:
        if isinstance(entry, PaymentRequirements):
            normalized.append(entry)
        elif isinstance(entry, dict):
            normalized.append(PaymentRequirements.from_wire(entry))
        else:
            raise TypeError(f"Unexpected accepts[] entry type: {type(entry).__name__}")

    for req in normalized:
        if req.scheme == "exact":
            return req
    schemes = ", ".join(r.scheme for r in normalized) or "<empty>"
    raise ValueError(
        f"No supported x402 scheme in proxy 402 response. "
        f"Accepted: {schemes}. This SDK supports: exact."
    )


def _assert_hex_address(value: str) -> str:
    if not _EVM_ADDRESS_RE.match(value):
        raise ValueError(f"Invalid EVM address: {value!r}")
    return value
