from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from eth_account import Account
from eth_account.signers.local import LocalAccount


@dataclass(frozen=True)
class SigComponents:
    v: int
    r: str
    s: str


def derive_address(private_key: str) -> str:
    acct: LocalAccount = Account.from_key(private_key)
    return acct.address


async def sign_typed_data(
    typed_data: dict[str, Any],
    message_key: str,
    private_key: str,
) -> SigComponents:
    all_types: dict[str, list[dict[str, str]]] = typed_data["types"]
    domain = typed_data["domain"]
    message = typed_data[message_key]

    primary_type = _primary_type_for(message_key, all_types)

    # Filter types to only EIP712Domain + the primary type (and its deps).
    # eth-account auto-detects primaryType by finding the type that is not
    # referenced by any other type; when multiple top-level types are present
    # the detection fails with "Unable to determine primary type".
    filtered_types = _collect_types(primary_type, all_types)

    signed = Account.sign_typed_data(
        private_key=private_key,
        full_message={
            "types": filtered_types,
            "domain": domain,
            "primaryType": primary_type,
            "message": message,
        },
    )

    r_val = signed.r
    s_val = signed.s
    r_hex = "0x" + (r_val.to_bytes(32, "big").hex() if isinstance(r_val, int) else r_val.hex())
    s_hex = "0x" + (s_val.to_bytes(32, "big").hex() if isinstance(s_val, int) else s_val.hex())
    return SigComponents(v=signed.v, r=r_hex, s=s_hex)


def _primary_type_for(message_key: str, types: dict[str, Any]) -> str:
    candidate = message_key[0].upper() + message_key[1:]
    if candidate in types:
        return candidate
    for name in types:
        if name != "EIP712Domain":
            return name
    raise ValueError(f"Cannot determine primary type for message key '{message_key}'")


def _collect_types(
    primary_type: str,
    all_types: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    """Return EIP712Domain plus primary_type and any types it references."""
    result: dict[str, list[dict[str, str]]] = {}
    if "EIP712Domain" in all_types:
        result["EIP712Domain"] = all_types["EIP712Domain"]

    visited: set[str] = set()
    queue = [primary_type]
    while queue:
        current = queue.pop()
        if current in visited or current not in all_types:
            continue
        visited.add(current)
        fields = all_types[current]
        result[current] = fields
        for field in fields:
            field_type = field["type"].rstrip("[]")
            if field_type in all_types and field_type not in visited:
                queue.append(field_type)

    return result
