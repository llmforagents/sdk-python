import pytest
import respx
import httpx
from eth_account import Account
from llm4agents.transport.http import HttpTransport
from llm4agents.transfer.transfer import Transfer
from llm4agents.transfer.types import QuoteResult, TransferResult
from llm4agents.transfer.signer import sign_typed_data, derive_address


MOCK_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
# Valid EVM addresses derived from well-known Hardhat test keys
ADDR_FROM = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
ADDR_TO = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
ADDR_SPENDER = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
ADDR_TOKEN = "0x2279B7A0a67DB372996a5FaB50D91eAA73d2eBe6"


@pytest.fixture
def transfer():
    return Transfer(HttpTransport("https://api.example.com", "test-key", 5.0))


def _mock_quote() -> dict:
    return {
        "fee": "1000000",
        "fee_formatted": "1.00",
        "fee_decimal": "1.0",
        "chain": "polygon",
        "chain_id": 137,
        "token": "USDC",
        "token_address": ADDR_TOKEN,
        "from": ADDR_FROM,
        "to": ADDR_TO,
        "amount": "10.00",
        "amount_base_units": "10000000",
        "deadline": 9999999999,
        "nonces": {"permit": 0, "transfer": 0},
        "typed_data": {
            "domain": {"name": "USDC", "version": "2", "chainId": 137, "verifyingContract": ADDR_TOKEN},
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Permit": [
                    {"name": "owner", "type": "address"},
                    {"name": "spender", "type": "address"},
                    {"name": "value", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "deadline", "type": "uint256"},
                ],
                "TransferPermit": [
                    {"name": "from", "type": "address"},
                    {"name": "to", "type": "address"},
                    {"name": "value", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "deadline", "type": "uint256"},
                ],
            },
            "permit": {
                "owner": ADDR_FROM,
                "spender": ADDR_SPENDER,
                "value": 10000000,
                "nonce": 0,
                "deadline": 9999999999,
            },
            "transferPermit": {
                "from": ADDR_FROM,
                "to": ADDR_TO,
                "value": 10000000,
                "nonce": 0,
                "deadline": 9999999999,
            },
        },
        "request_id": "req-quote",
    }


@respx.mock
async def test_quote(transfer):
    respx.post("https://api.example.com/api/v1/transfer/quote").mock(
        return_value=httpx.Response(200, json=_mock_quote())
    )
    result = await transfer.quote({
        "chain": "polygon", "token": "USDC",
        "from": ADDR_FROM, "to": ADDR_TO, "amount": "10.00",
    })
    assert isinstance(result, QuoteResult)
    assert result.chain == "polygon"
    assert result.fee == "1000000"


@respx.mock
async def test_submit(transfer):
    respx.post("https://api.example.com/api/v1/transfer/submit").mock(
        return_value=httpx.Response(200, json={
            "tx_hash": "0xhash",
            "explorer_url": "https://polygonscan.com/tx/0xhash",
            "from": ADDR_FROM,
            "to": ADDR_TO,
            "chain": "polygon",
            "token": "USDC",
            "amount": "10.00",
            "fee": "1.00",
            "request_id": "req-submit",
        })
    )
    quote_data = _mock_quote()
    quote = QuoteResult.from_dict(quote_data)
    result = await transfer.submit(quote, MOCK_KEY)
    assert isinstance(result, TransferResult)
    assert result.tx_hash == "0xhash"


async def test_derive_address():
    addr = derive_address(MOCK_KEY)
    expected = Account.from_key(MOCK_KEY).address
    assert addr == expected


def test_sign_typed_data_returns_components():
    quote_data = _mock_quote()
    td = quote_data["typed_data"]
    sig = sign_typed_data(td, "permit", MOCK_KEY)
    assert sig.v in (27, 28)
    assert sig.r.startswith("0x")
    assert sig.s.startswith("0x")
    assert len(sig.r) == 66
    assert len(sig.s) == 66
