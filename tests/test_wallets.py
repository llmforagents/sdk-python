import pytest
import respx
import httpx
from llm4agents.transport.http import HttpTransport
from llm4agents.wallets.wallets import Wallets
from llm4agents.wallets.types import WalletInfo, Balance, WalletBalance, Transaction, TransactionList


@pytest.fixture
def wallets():
    return Wallets(HttpTransport("https://api.example.com", "test-key", 5.0))


@respx.mock
async def test_generate(wallets):
    respx.post("https://api.example.com/api/v1/wallets/generate").mock(
        return_value=httpx.Response(200, json={
            "chain": "polygon",
            "token": "USDC",
            "address": "0xabc",
            "created_at": "2024-01-01T00:00:00Z",
            "request_id": "req-1",
        })
    )
    info = await wallets.generate({"chain": "polygon", "token": "USDC"})
    assert isinstance(info, WalletInfo)
    assert info.chain == "polygon"
    assert info.address == "0xabc"
    assert info.request_id == "req-1"


@respx.mock
async def test_balance(wallets):
    respx.get("https://api.example.com/api/v1/balance").mock(
        return_value=httpx.Response(200, json={
            "uuid": "agent-uuid",
            "available_usd_cents": 500,
            "available_usd": "5.00",
            "total_deposited_usd": "10.00",
            "total_spent_usd": "5.00",
            "wallets": [
                {
                    "chain": "polygon",
                    "token": "USDC",
                    "address": "0xabc",
                    "balance_usd_cents": 500,
                    "balance_usd": "5.00",
                }
            ],
        })
    )
    bal = await wallets.balance()
    assert isinstance(bal, Balance)
    assert bal.available_usd_cents == 500
    assert len(bal.wallets) == 1
    assert isinstance(bal.wallets[0], WalletBalance)


@respx.mock
async def test_transactions(wallets):
    respx.get("https://api.example.com/api/v1/transactions").mock(
        return_value=httpx.Response(200, json={
            "transactions": [
                {
                    "id": "tx-1",
                    "type": "deposit",
                    "amount_usd_cents": 1000,
                    "amount_usd": "10.00",
                    "created_at": "2024-01-01T00:00:00Z",
                    "description": "USDC deposit",
                }
            ],
            "limit": 50,
            "offset": 0,
            "total": 1,
            "request_id": "req-2",
        })
    )
    result = await wallets.transactions()
    assert isinstance(result, TransactionList)
    assert result.total == 1
    assert isinstance(result.transactions[0], Transaction)
    assert result.transactions[0].type == "deposit"
