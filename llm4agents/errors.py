from typing import Literal

ErrorCode = Literal[
    "auth_error",
    "rate_limited",
    "network_error",
    "timeout",
    "api_error",
    "model_not_found",
    "model_disabled",
    "context_overflow",
    "insufficient_balance",
    "gas_spike",
    "signature_mismatch",
    "invalid_token",
    "operator_unavailable",
    "deadline_expired",
    "tool_not_found",
    "tool_execution_error",
    "tool_loop_limit",
]

_STATUS_TO_CODE: dict[int, str] = {
    401: "auth_error",
    403: "auth_error",
    429: "rate_limited",
}

_BODY_CODE_ALLOWLIST: frozenset[str] = frozenset([
    "model_not_found",
    "model_disabled",
    "context_overflow",
    "insufficient_balance",
    "gas_spike",
    "signature_mismatch",
    "invalid_token",
    "operator_unavailable",
    "deadline_expired",
    "tool_not_found",
    "tool_execution_error",
    "tool_loop_limit",
])


class LLM4AgentsError(Exception):
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int | None,
        request_id: str | None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.request_id = request_id

    def __str__(self) -> str:
        return self.message


def map_http_error(
    status: int,
    body: dict,  # type: ignore[type-arg]
    request_id: str | None = None,
) -> LLM4AgentsError:
    body_code: str | None = None
    if isinstance(body.get("error"), dict):
        candidate = body["error"].get("code")
        if candidate in _BODY_CODE_ALLOWLIST:
            body_code = candidate

    code = body_code or _STATUS_TO_CODE.get(status, "api_error")

    try:
        message = body.get("error", {}).get("message") or body.get("message") or f"HTTP {status}"
    except AttributeError:
        message = f"HTTP {status}"

    return LLM4AgentsError(message, code, status, request_id)
