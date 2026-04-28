import pytest
from llm4agents.errors import LLM4AgentsError, map_http_error


def test_error_attributes():
    err = LLM4AgentsError("bad auth", "auth_error", 401, "req-123")
    assert err.message == "bad auth"
    assert err.code == "auth_error"
    assert err.status_code == 401
    assert err.request_id == "req-123"
    assert "bad auth" in str(err)


def test_map_http_error_401():
    err = map_http_error(401, {}, "req-1")
    assert err.code == "auth_error"
    assert err.status_code == 401


def test_map_http_error_429():
    err = map_http_error(429, {}, "req-2")
    assert err.code == "rate_limited"


def test_map_http_error_with_code_in_body():
    err = map_http_error(422, {"error": {"code": "model_disabled"}}, "req-3")
    assert err.code == "model_disabled"


def test_map_http_error_500():
    err = map_http_error(500, {}, None)
    assert err.code == "api_error"
    assert err.status_code == 500
    assert err.request_id is None
