from __future__ import annotations
from typing import Any
from llm4agents.transport.http import HttpTransport
from llm4agents.wallets.types import WalletInfo, Balance, TransactionList


class Wallets:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def generate(self, params: dict[str, str]) -> WalletInfo:
        data = await self._http.post("/api/v1/wallets/generate", params)
        return WalletInfo.from_dict(data)

    async def balance(self) -> Balance:
        data = await self._http.get("/api/v1/balance")
        return Balance.from_dict(data)

    async def transactions(
        self,
        filter: dict[str, Any] | None = None,
    ) -> TransactionList:
        params: dict[str, Any] = {}
        if filter:
            params.update(filter)
        data = await self._http.get("/api/v1/transactions", params=params or None)
        return TransactionList.from_dict(data)
