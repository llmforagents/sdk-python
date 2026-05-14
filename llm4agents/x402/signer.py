"""Signer abstraction + ``eth_account`` adapter for x402 EIP-3009 signatures.

Two ways to bring a wallet:

  1. Pass an ``eth_account.LocalAccount`` (e.g. from
     ``Account.from_key(...)``) via ``eth_account_to_signer(account)``.
     This is the easiest path because ``eth_account`` is already a
     transitive dep of the SDK (used by gasless transfers).

  2. Provide your own ``Signer`` implementation. Useful when wrapping
     a hardware wallet, KMS-backed key, or any other signing primitive.

Both produce the same ``Signer`` shape the rest of this module consumes,
so swapping wallets at the boundary doesn't ripple. Ports & Adapters:
the SDK depends on the ``Signer`` Protocol in ``types.py``, not on
``eth_account``.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

from llm4agents.x402.types import (
    CHAIN_ID_BY_NETWORK,
    USDC_ADDRESS_BY_NETWORK,
    USDC_DOMAIN_NAME_BY_NETWORK,
    X402_CAIP2_BY_NETWORK,
    Signer,
    X402Network,
)

# EIP-3009 ``TransferWithAuthorization`` typed-data types. Note that
# ``eth_account.messages.encode_typed_data(full_message=...)`` requires
# ``EIP712Domain`` to be included in ``types`` — viem auto-adds it, but
# the Python eth_account library does not.
TRANSFER_WITH_AUTHORIZATION_TYPES: dict[str, list[dict[str, str]]] = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "TransferWithAuthorization": [
        {"name": "from", "type": "address"},
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "validAfter", "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce", "type": "bytes32"},
    ],
}


@dataclass(frozen=True)
class _EthAccountSigner:
    """Adapter wrapping an ``eth_account.LocalAccount`` as a ``Signer``."""

    address: str
    _private_key: str  # 0x-prefixed hex, kept for sign_typed_data

    def sign_typed_data(
        self,
        *,
        domain: dict[str, Any],
        types: dict[str, list[dict[str, str]]],
        primary_type: str,
        message: dict[str, Any],
    ) -> str:
        # Lazy import — eth_account is a hard dep of the SDK (gasless
        # transfers depend on it) but isolating the import keeps the
        # x402.types module light for users who only need the Signer
        # protocol and a custom adapter.
        from eth_account import Account
        from eth_account.messages import encode_typed_data

        # eth_account expects EIP-3009 message values to be integers
        # (uint256). The wire shape carries strings, so we coerce here.
        normalized = _normalize_eip3009_message(message)
        full_message = {
            "domain": domain,
            "types": types,
            "primaryType": primary_type,
            "message": normalized,
        }
        encoded = encode_typed_data(full_message=full_message)
        signed = Account.sign_message(encoded, private_key=self._private_key)
        sig_hex = signed.signature.hex()
        return sig_hex if sig_hex.startswith("0x") else f"0x{sig_hex}"


def eth_account_to_signer(account: Any) -> Signer:
    """Convert an ``eth_account.LocalAccount`` to the SDK's ``Signer`` shape.

    Accepts any object exposing ``.address`` (checksum-cased) and ``.key``
    (a ``HexBytes``, ``bytes``, or ``str``). The duck-typing keeps this
    function from a hard ``eth_account`` import at module load time.
    """
    address = getattr(account, "address", None)
    if not isinstance(address, str) or not address.startswith("0x"):
        raise ValueError(
            "eth_account_to_signer: expected an eth_account.LocalAccount "
            "with a .address attribute. Got: " + repr(account)
        )
    key = getattr(account, "key", None)
    if key is None:
        raise ValueError(
            "eth_account_to_signer: account is missing a .key attribute "
            "(needed for signing). For external signers, implement the "
            "Signer protocol directly instead."
        )
    if hasattr(key, "hex"):
        key_hex = key.hex()
    elif isinstance(key, bytes):
        key_hex = key.hex()
    else:
        key_hex = str(key)
    if not key_hex.startswith("0x"):
        key_hex = f"0x{key_hex}"
    return _EthAccountSigner(address=address, _private_key=key_hex)


def build_transfer_with_authorization_typed_data(
    *,
    signer: Signer,
    network: X402Network,
    to: str,
    value: str,
    valid_after: str,
    valid_before: str,
    nonce: str,
) -> dict[str, Any]:
    """Build the EIP-3009 ``TransferWithAuthorization`` typed-data payload.

    Constants per network come from the proxy's on-chain-verified config
    (``USD Coin`` on mainnet vs ``USDC`` on Sepolia — a real gotcha that
    breaks signatures if copy-pasted from the spec example).
    """
    return {
        "domain": {
            "name": USDC_DOMAIN_NAME_BY_NETWORK[network],
            "version": "2",
            "chainId": CHAIN_ID_BY_NETWORK[network],
            "verifyingContract": USDC_ADDRESS_BY_NETWORK[network],
        },
        "types": TRANSFER_WITH_AUTHORIZATION_TYPES,
        "primaryType": "TransferWithAuthorization",
        "message": {
            "from": signer.address,
            "to": to,
            "value": value,
            "validAfter": valid_after,
            "validBefore": valid_before,
            "nonce": nonce,
        },
    }


def generate_nonce() -> str:
    """Generate a fresh 32-byte hex nonce using ``secrets.token_hex``."""
    return "0x" + secrets.token_hex(32)


def network_to_caip2(network: X402Network) -> str:
    """Resolve the CAIP-2 network identifier the proxy emits in
    ``PaymentRequirements.network`` (e.g. ``eip155:8453``)."""
    return X402_CAIP2_BY_NETWORK[network]


def _normalize_eip3009_message(message: dict[str, Any]) -> dict[str, Any]:
    """Coerce wire-shape EIP-3009 fields into integers for eth_account.

    eth_account's typed-data encoder demands uint256 fields as Python
    ints; the x402 wire format ships them as decimal strings.
    """
    out: dict[str, Any] = dict(message)
    for k in ("value", "validAfter", "validBefore"):
        if k in out and isinstance(out[k], str):
            out[k] = int(out[k])
    # nonce stays as bytes32 — eth_account accepts a 0x-prefixed hex
    # string for bytes32 fields.
    return out
