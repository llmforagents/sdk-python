from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuoteResult:
    fee: str
    fee_formatted: str
    fee_decimal: str
    chain: str
    chain_id: int
    token: str
    token_address: str
    from_address: str
    to_address: str
    amount: str
    amount_base_units: str
    deadline: int
    nonces: dict[str, int]
    typed_data: dict[str, Any]
    request_id: str
    forwarder_address: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuoteResult:
        return cls(
            fee=d["fee"],
            fee_formatted=d["fee_formatted"],
            fee_decimal=d["fee_decimal"],
            chain=d["chain"],
            chain_id=d["chain_id"],
            token=d["token"],
            token_address=d["token_address"],
            from_address=d["from"],
            to_address=d["to"],
            amount=d["amount"],
            amount_base_units=d["amount_base_units"],
            deadline=d["deadline"],
            nonces=d["nonces"],
            typed_data=d["typed_data"],
            request_id=d["request_id"],
            forwarder_address=d.get("forwarderAddress", ""),
        )


@dataclass(frozen=True)
class TransferResult:
    tx_hash: str
    explorer_url: str
    from_address: str
    to_address: str
    chain: str
    token: str
    amount: str
    fee: str
    request_id: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TransferResult:
        return cls(
            tx_hash=d["tx_hash"],
            explorer_url=d["explorer_url"],
            from_address=d["from"],
            to_address=d["to"],
            chain=d["chain"],
            token=d["token"],
            amount=d["amount"],
            fee=d["fee"],
            request_id=d["request_id"],
        )
