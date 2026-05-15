"""Tests for the x402 signer adapter + typed-data builder."""
from __future__ import annotations

import re

import pytest
from eth_account import Account

from llm4agents.x402.signer import (
    TRANSFER_WITH_AUTHORIZATION_TYPES,
    build_transfer_with_authorization_typed_data,
    eth_account_to_signer,
    generate_nonce,
    network_to_caip2,
)
from llm4agents.x402.types import (
    CHAIN_ID_BY_NETWORK,
    USDC_ADDRESS_BY_NETWORK,
    USDC_DOMAIN_NAME_BY_NETWORK,
)

TEST_KEY = "0x" + "1" * 64


def test_eth_account_to_signer_exposes_address_and_sign() -> None:
    acct = Account.from_key(TEST_KEY)
    signer = eth_account_to_signer(acct)
    assert signer.address == acct.address
    assert callable(signer.sign_typed_data)


def test_eth_account_to_signer_rejects_missing_address() -> None:
    class Junk:
        key = b"0" * 32

    with pytest.raises(ValueError, match="expected an eth_account"):
        eth_account_to_signer(Junk())


def test_eth_account_to_signer_rejects_missing_key() -> None:
    class NoKey:
        address = "0x" + "1" * 40

    with pytest.raises(ValueError, match="missing a .key attribute"):
        eth_account_to_signer(NoKey())


def test_generate_nonce_format() -> None:
    nonce = generate_nonce()
    # 0x + 64 hex chars = 66 total = 32-byte bytes32
    assert re.match(r"^0x[0-9a-f]{64}$", nonce), nonce


def test_generate_nonce_uniqueness() -> None:
    nonces = {generate_nonce() for _ in range(100)}
    assert len(nonces) == 100, "nonces collided — token_hex is broken"


def test_network_to_caip2() -> None:
    assert network_to_caip2("base") == "eip155:8453"
    assert network_to_caip2("base-sepolia") == "eip155:84532"


def test_build_typed_data_mainnet_uses_usd_coin_name() -> None:
    acct = Account.from_key(TEST_KEY)
    signer = eth_account_to_signer(acct)
    td = build_transfer_with_authorization_typed_data(
        signer=signer,
        network="base",
        to="0x" + "2" * 40,
        value="100",
        valid_after="0",
        valid_before="2000000000",
        nonce="0x" + "a" * 64,
    )
    assert td["domain"]["name"] == "USD Coin"
    assert td["domain"]["chainId"] == 8453
    assert td["domain"]["verifyingContract"] == USDC_ADDRESS_BY_NETWORK["base"]
    assert td["primaryType"] == "TransferWithAuthorization"


def test_build_typed_data_sepolia_uses_usdc_name() -> None:
    acct = Account.from_key(TEST_KEY)
    signer = eth_account_to_signer(acct)
    td = build_transfer_with_authorization_typed_data(
        signer=signer,
        network="base-sepolia",
        to="0x" + "2" * 40,
        value="100",
        valid_after="0",
        valid_before="2000000000",
        nonce="0x" + "a" * 64,
    )
    assert td["domain"]["name"] == "USDC"
    assert td["domain"]["chainId"] == 84532


def test_build_typed_data_message_carries_signer_address() -> None:
    acct = Account.from_key(TEST_KEY)
    signer = eth_account_to_signer(acct)
    td = build_transfer_with_authorization_typed_data(
        signer=signer,
        network="base",
        to="0x" + "2" * 40,
        value="123",
        valid_after="100",
        valid_before="2000000000",
        nonce="0x" + "b" * 64,
    )
    msg = td["message"]
    assert msg["from"] == acct.address
    assert msg["to"] == "0x" + "2" * 40
    assert msg["value"] == "123"
    assert msg["validAfter"] == "100"
    assert msg["validBefore"] == "2000000000"
    assert msg["nonce"] == "0x" + "b" * 64


def test_signer_signature_is_deterministic_for_same_inputs() -> None:
    """EIP-712 signing on a fixed key + fixed message must be deterministic."""
    acct = Account.from_key(TEST_KEY)
    signer = eth_account_to_signer(acct)
    domain = {
        "name": USDC_DOMAIN_NAME_BY_NETWORK["base-sepolia"],
        "version": "2",
        "chainId": CHAIN_ID_BY_NETWORK["base-sepolia"],
        "verifyingContract": USDC_ADDRESS_BY_NETWORK["base-sepolia"],
    }
    msg = {
        "from": signer.address,
        "to": "0x" + "2" * 40,
        "value": "100",
        "validAfter": "0",
        "validBefore": "2000000000",
        "nonce": "0x" + "c" * 64,
    }
    sig1 = signer.sign_typed_data(
        domain=domain,
        types=TRANSFER_WITH_AUTHORIZATION_TYPES,
        primary_type="TransferWithAuthorization",
        message=msg,
    )
    sig2 = signer.sign_typed_data(
        domain=domain,
        types=TRANSFER_WITH_AUTHORIZATION_TYPES,
        primary_type="TransferWithAuthorization",
        message=msg,
    )
    assert sig1 == sig2
    assert sig1.startswith("0x")
    assert len(sig1) == 132  # 0x + 65 bytes * 2 hex chars


def test_signer_signature_changes_with_different_message() -> None:
    acct = Account.from_key(TEST_KEY)
    signer = eth_account_to_signer(acct)
    domain = {
        "name": "USDC",
        "version": "2",
        "chainId": 84532,
        "verifyingContract": USDC_ADDRESS_BY_NETWORK["base-sepolia"],
    }
    base = {
        "from": signer.address,
        "to": "0x" + "2" * 40,
        "value": "100",
        "validAfter": "0",
        "validBefore": "2000000000",
        "nonce": "0x" + "c" * 64,
    }
    sig_a = signer.sign_typed_data(
        domain=domain,
        types=TRANSFER_WITH_AUTHORIZATION_TYPES,
        primary_type="TransferWithAuthorization",
        message=base,
    )
    different = {**base, "value": "200"}
    sig_b = signer.sign_typed_data(
        domain=domain,
        types=TRANSFER_WITH_AUTHORIZATION_TYPES,
        primary_type="TransferWithAuthorization",
        message=different,
    )
    assert sig_a != sig_b
