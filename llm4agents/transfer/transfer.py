from __future__ import annotations
from typing import Any
from llm4agents.transport.http import HttpTransport
from llm4agents.transfer.types import QuoteResult, TransferResult
from llm4agents.transfer.signer import sign_typed_data, derive_address


class Transfer:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def quote(self, params: dict[str, Any]) -> QuoteResult:
        data = await self._http.post("/api/v1/transfer/quote", params)
        return QuoteResult.from_dict(data)

    async def submit(self, quote: QuoteResult, private_key: str) -> TransferResult:
        permit_sig = sign_typed_data(quote.typed_data, "permit", private_key)
        transfer_sig = sign_typed_data(quote.typed_data, "transferPermit", private_key)

        payload: dict[str, Any] = {
            "quote_request_id": quote.request_id,
            "permit_signature": {
                "v": permit_sig.v,
                "r": permit_sig.r,
                "s": permit_sig.s,
            },
            "transfer_signature": {
                "v": transfer_sig.v,
                "r": transfer_sig.r,
                "s": transfer_sig.s,
            },
        }
        data = await self._http.post("/api/v1/transfer/submit", payload)
        return TransferResult.from_dict(data)

    async def send(self, params: dict[str, Any]) -> TransferResult:
        private_key: str = params["private_key"]
        from_address = derive_address(private_key)
        quote_params: dict[str, Any] = {
            "chain": params["chain"],
            "token": params["token"],
            "from": from_address,
            "to": params["to"],
            "amount": params["amount"],
        }
        quote = await self.quote(quote_params)
        return await self.submit(quote, private_key)
