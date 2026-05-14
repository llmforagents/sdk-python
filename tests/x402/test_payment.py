"""Tests for the x402 payment-payload helpers."""
from __future__ import annotations

import base64
import json

import pytest
from eth_account import Account

from llm4agents.x402.payment import (
    decode_payment_required_header,
    encode_payment_header,
    pick_supported_requirements,
    sign_from_requirements,
)
from llm4agents.x402.signer import eth_account_to_signer
from llm4agents.x402.types import PaymentPayload, PaymentRequirements

TEST_KEY = "0x" + "1" * 64


def _make_requirements() -> PaymentRequirements:
    return PaymentRequirements(
        scheme="exact",
        network="eip155:84532",
        max_amount_required="10000",
        asset="0x036CbD53842c5426634e7929541eC2318f3dCF7e",
        pay_to="0x0000000000000000000000000000000000000033",
        max_timeout_seconds=60,
        extra={"name": "USDC", "version": "2"},
    )


@pytest.mark.asyncio
async def test_sign_from_requirements_happy_path() -> None:
    signer = eth_account_to_signer(Account.from_key(TEST_KEY))
    req = _make_requirements()
    signed = await sign_from_requirements(
        signer=signer, network="base-sepolia", requirements=req
    )

    assert signed.payment_payload.scheme == "exact"
    assert signed.payment_payload.network == "eip155:84532"
    assert signed.payment_payload.authorization_value == "10000"
    assert signed.payment_payload.authorization_to == req.pay_to
    assert signed.payment_payload.authorization_from == signer.address
    assert signed.payment_payload.signature.startswith("0x")
    assert len(signed.payment_payload.signature) == 132


@pytest.mark.asyncio
async def test_sign_from_requirements_honors_recipient_override() -> None:
    signer = eth_account_to_signer(Account.from_key(TEST_KEY))
    req = _make_requirements()
    override = "0x" + "f" * 40
    signed = await sign_from_requirements(
        signer=signer,
        network="base-sepolia",
        requirements=req,
        recipient_override=override,
    )
    assert signed.payment_payload.authorization_to == override


@pytest.mark.asyncio
async def test_sign_from_requirements_rejects_malformed_address() -> None:
    signer = eth_account_to_signer(Account.from_key(TEST_KEY))
    bad = PaymentRequirements(
        scheme="exact",
        network="eip155:84532",
        max_amount_required="1",
        asset="0x036CbD53842c5426634e7929541eC2318f3dCF7e",
        pay_to="not-an-address",
        max_timeout_seconds=60,
    )
    with pytest.raises(ValueError, match="Invalid EVM address"):
        await sign_from_requirements(signer=signer, network="base-sepolia", requirements=bad)


@pytest.mark.asyncio
async def test_sign_from_requirements_emits_unique_nonces_across_calls() -> None:
    signer = eth_account_to_signer(Account.from_key(TEST_KEY))
    req = _make_requirements()
    a = await sign_from_requirements(signer=signer, network="base-sepolia", requirements=req)
    b = await sign_from_requirements(signer=signer, network="base-sepolia", requirements=req)
    assert a.payment_payload.authorization_nonce != b.payment_payload.authorization_nonce


def test_encode_payment_header_round_trip() -> None:
    payload = PaymentPayload(
        x402_version=1,
        scheme="exact",
        network="eip155:8453",
        signature="0x" + "a" * 130,
        authorization_from="0x" + "1" * 40,
        authorization_to="0x" + "2" * 40,
        authorization_value="1",
        authorization_valid_after="0",
        authorization_valid_before="2000000000",
        authorization_nonce="0x" + "b" * 64,
    )
    encoded = encode_payment_header(payload)
    # Decodes to JSON with wire shape
    decoded = json.loads(base64.b64decode(encoded).decode("utf-8"))
    assert decoded["scheme"] == "exact"
    assert decoded["network"] == "eip155:8453"
    assert decoded["x402Version"] == 1
    assert decoded["payload"]["signature"] == payload.signature
    assert decoded["payload"]["authorization"]["from"] == payload.authorization_from


def test_decode_payment_required_header_parses_accepts() -> None:
    raw = {
        "x402Version": 2,
        "accepts": [
            {
                "scheme": "exact",
                "network": "eip155:84532",
                "maxAmountRequired": "10000",
                "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                "payTo": "0x" + "1" * 40,
                "maxTimeoutSeconds": 60,
            }
        ],
    }
    header = base64.b64encode(json.dumps(raw).encode("utf-8")).decode("ascii")
    version, accepts = decode_payment_required_header(header)
    assert version == 2
    assert len(accepts) == 1
    assert accepts[0].scheme == "exact"
    assert accepts[0].max_amount_required == "10000"


def test_decode_payment_required_header_rejects_missing_version() -> None:
    raw = {"accepts": []}
    header = base64.b64encode(json.dumps(raw).encode("utf-8")).decode("ascii")
    with pytest.raises(ValueError, match="x402Version"):
        decode_payment_required_header(header)


def test_decode_payment_required_header_rejects_non_array_accepts() -> None:
    raw = {"x402Version": 1, "accepts": "not-an-array"}
    header = base64.b64encode(json.dumps(raw).encode("utf-8")).decode("ascii")
    with pytest.raises(ValueError, match="accepts"):
        decode_payment_required_header(header)


def test_pick_supported_requirements_picks_exact() -> None:
    upto_req = PaymentRequirements(
        scheme="upto",
        network="eip155:8453",
        max_amount_required="500",
        asset="0x" + "0" * 40,
        pay_to="0x" + "1" * 40,
        max_timeout_seconds=60,
    )
    exact_req = PaymentRequirements(
        scheme="exact",
        network="eip155:8453",
        max_amount_required="100",
        asset="0x" + "0" * 40,
        pay_to="0x" + "1" * 40,
        max_timeout_seconds=60,
    )
    picked = pick_supported_requirements([upto_req, exact_req])
    assert picked.scheme == "exact"
    assert picked.max_amount_required == "100"


def test_pick_supported_requirements_raises_when_no_exact() -> None:
    upto_req = PaymentRequirements(
        scheme="upto",
        network="eip155:8453",
        max_amount_required="500",
        asset="0x" + "0" * 40,
        pay_to="0x" + "1" * 40,
        max_timeout_seconds=60,
    )
    with pytest.raises(ValueError, match="No supported x402 scheme"):
        pick_supported_requirements([upto_req])


def test_pick_supported_requirements_accepts_wire_dicts() -> None:
    accepts = [
        {
            "scheme": "exact",
            "network": "eip155:8453",
            "maxAmountRequired": "100",
            "asset": "0x" + "0" * 40,
            "payTo": "0x" + "1" * 40,
            "maxTimeoutSeconds": 60,
        }
    ]
    picked = pick_supported_requirements(accepts)
    assert picked.scheme == "exact"
    assert picked.max_amount_required == "100"
