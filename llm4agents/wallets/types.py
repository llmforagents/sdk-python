from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WalletInfo:
    chain: str
    token: str
    address: str
    created_at: str
    request_id: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WalletInfo:
        return cls(
            chain=d["chain"],
            token=d["token"],
            address=d["address"],
            created_at=d["created_at"],
            request_id=d["request_id"],
        )


@dataclass(frozen=True)
class WalletBalance:
    chain: str
    token: str
    address: str
    balance_usd_cents: int
    balance_usd: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WalletBalance:
        return cls(
            chain=d["chain"],
            token=d["token"],
            address=d["address"],
            balance_usd_cents=d["balance_usd_cents"],
            balance_usd=d["balance_usd"],
        )


@dataclass(frozen=True)
class Balance:
    uuid: str
    available_usd_cents: int
    available_usd: str
    total_deposited_usd: str
    total_spent_usd: str
    wallets: tuple[WalletBalance, ...]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Balance:
        return cls(
            uuid=d["uuid"],
            available_usd_cents=d["available_usd_cents"],
            available_usd=d["available_usd"],
            total_deposited_usd=d["total_deposited_usd"],
            total_spent_usd=d["total_spent_usd"],
            wallets=tuple(WalletBalance.from_dict(w) for w in d.get("wallets", [])),
        )


@dataclass(frozen=True)
class Transaction:
    id: str
    type: str
    amount_usd_cents: int
    amount_usd: str
    created_at: str
    description: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transaction:
        return cls(
            id=d["id"],
            type=d["type"],
            amount_usd_cents=d["amount_usd_cents"],
            amount_usd=d["amount_usd"],
            created_at=d["created_at"],
            description=d.get("description", ""),
        )


@dataclass(frozen=True)
class TransactionList:
    transactions: tuple[Transaction, ...]
    limit: int
    offset: int
    total: int
    request_id: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TransactionList:
        return cls(
            transactions=tuple(Transaction.from_dict(t) for t in d.get("transactions", [])),
            limit=d["limit"],
            offset=d["offset"],
            total=d["total"],
            request_id=d["request_id"],
        )
