from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Any
import json
import httpx
from llm4agents.errors import LLM4AgentsError, map_http_error
from llm4agents.x402.payment import (
    decode_payment_required_header,
    pick_supported_requirements,
    sign_from_requirements,
)
from llm4agents.x402.types import (
    PaymentConfig,
    PaymentRequirements,
    X402Network,
    X402PaymentRequiredError,
)

# Routes that the proxy currently accepts x402 payment on.
# Routes that accept x402 walk-up payment. Browser sessions
# (``session_*`` MCP tools) are intentionally excluded — they stay
# Bearer-only because the launch + per-30s + per-action billing model
# is incompatible with a single per-call signed authorization.
_X402_ALLOWED_PATHS: frozenset[str] = frozenset({
    "/v1/chat/completions",
    "/v1/scrape/fetch_html",
    "/v1/scrape/markdown",
    "/v1/scrape/links",
    "/v1/scrape/screenshot",
    "/v1/scrape/pdf",
    "/v1/scrape/extract",
    "/v1/search/google",
    "/v1/search/news",
    "/v1/search/maps",
    "/v1/search/batch",
    "/v1/image/generate",
    "/v1/image/edit",
    "/v1/image/analyze",
})


class HttpTransport:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float,
        *,
        payment: PaymentConfig | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._payment: PaymentConfig = payment or PaymentConfig(mode="bearer")

    def _client(self, *, with_auth: bool = True) -> httpx.AsyncClient:
        """Open an httpx client.

        ``with_auth=False`` returns a client with NO ``Authorization``
        header — used by the x402 probe (unauthenticated POST).
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if with_auth and self._payment.mode == "bearer":
            headers["Authorization"] = f"Bearer {self._api_key}"
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout,
            http2=True,
        )

    @property
    def _x402_network(self) -> X402Network:
        if self._payment.mode == "x402" and self._payment.network is not None:
            return self._payment.network
        return "base"

    def _request_id(self, response: httpx.Response) -> str | None:
        return response.headers.get("x-request-id")

    async def _resolve_x402_payment_header(
        self, path: str, body: dict[str, Any]
    ) -> str:
        """In x402 mode: probe the path and produce the ``X-PAYMENT`` header.

        Raises ``LLM4AgentsError`` if the path is not in the proxy's x402
        allowlist (currently only ``/v1/chat/completions``).
        """
        if path not in _X402_ALLOWED_PATHS:
            allowed = ", ".join(sorted(_X402_ALLOWED_PATHS))
            raise LLM4AgentsError(
                f"x402 mode is only available on {allowed}; cannot use it "
                f"on POST {path}. Instantiate a Bearer-mode client for "
                f"this endpoint.",
                "x402_payment_required",
                None,
                None,
            )
        requirements = await self._probe_for_requirements(path, body)
        if self._payment.signer is None:
            raise LLM4AgentsError(
                "x402 mode requires a signer in PaymentConfig",
                "x402_payment_required",
                None,
                None,
            )
        signed = await sign_from_requirements(
            signer=self._payment.signer,
            network=self._x402_network,
            requirements=requirements,
            recipient_override=self._payment.pay_to,
        )
        return signed.encoded_header

    async def _probe_for_requirements(
        self, path: str, body: dict[str, Any]
    ) -> PaymentRequirements:
        """Issue an unauthenticated POST to read live ``PaymentRequirements``."""
        async with self._client(with_auth=False) as client:
            try:
                res = await client.post(path, content=json.dumps(body))
            except httpx.TimeoutException as e:
                raise LLM4AgentsError(str(e), "timeout", None, None) from e
            except httpx.NetworkError as e:
                raise LLM4AgentsError(str(e), "network_error", None, None) from e

        if res.status_code != 402:
            text = res.text[:200] if res.text else ""
            raise LLM4AgentsError(
                f"x402 probe expected HTTP 402 but got {res.status_code}: {text}",
                "api_error",
                res.status_code,
                self._request_id(res),
            )

        header_value = res.headers.get("payment-required")
        if header_value is not None:
            _version, accepts = decode_payment_required_header(header_value)
            return pick_supported_requirements(accepts)

        try:
            parsed = res.json()
        except ValueError:
            raise LLM4AgentsError(
                "x402 probe: 402 with no PAYMENT-REQUIRED header and "
                "non-JSON body",
                "api_error",
                res.status_code,
                self._request_id(res),
            ) from None
        accepts_raw: Any = parsed.get("accepts") if isinstance(parsed, dict) else None
        if not isinstance(accepts_raw, list):
            raise LLM4AgentsError(
                "x402 probe: 402 body has no accepts[] array",
                "api_error",
                res.status_code,
                self._request_id(res),
            )
        return pick_supported_requirements(accepts_raw)

    def _maybe_throw_x402(self, res: httpx.Response, body_text: str) -> None:
        """If the 402 response carries an x402 paymentRequirements shape,
        raise the typed error. Otherwise return (caller falls through to
        the generic ``map_http_error`` path)."""
        request_id = self._request_id(res)
        header_value = res.headers.get("payment-required")
        if header_value is not None:
            version, accepts = decode_payment_required_header(header_value)
            raise X402PaymentRequiredError(
                body_text or "Payment required",
                accepts,
                version,
                res.status_code,
                request_id,
            )
        try:
            parsed = json.loads(body_text) if body_text else {}
        except (ValueError, json.JSONDecodeError):
            return
        if (
            isinstance(parsed, dict)
            and isinstance(parsed.get("accepts"), list)
            and isinstance(parsed.get("x402Version"), int)
        ):
            accepts_list = [
                PaymentRequirements.from_wire(r)
                for r in parsed["accepts"]
                if isinstance(r, dict)
            ]
            raise X402PaymentRequiredError(
                body_text,
                accepts_list,
                parsed["x402Version"],
                res.status_code,
                request_id,
            )

    async def _build_post_headers(
        self, path: str, body: dict[str, Any]
    ) -> dict[str, str]:
        """Returns the auth headers for a POST: either Bearer (default) or
        the freshly signed ``X-PAYMENT`` in x402 mode."""
        if self._payment.mode == "bearer":
            return {"Authorization": f"Bearer {self._api_key}"}
        header = await self._resolve_x402_payment_header(path, body)
        return {"X-PAYMENT": header}

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        async with self._client() as client:
            try:
                res = await client.get(path, params=params)
            except httpx.TimeoutException as e:
                raise LLM4AgentsError(str(e), "timeout", None, None) from e
            except httpx.NetworkError as e:
                raise LLM4AgentsError(str(e), "network_error", None, None) from e
            if res.status_code >= 400:
                raise map_http_error(res.status_code, res.json(), self._request_id(res))
            return res.json()

    async def post(self, path: str, body: dict[str, Any]) -> Any:
        # Build auth headers FIRST (in x402 mode this issues the probe).
        auth_headers = await self._build_post_headers(path, body)
        async with self._client(with_auth=False) as client:
            try:
                res = await client.post(
                    path, content=json.dumps(body), headers=auth_headers
                )
            except httpx.TimeoutException as e:
                raise LLM4AgentsError(str(e), "timeout", None, None) from e
            except httpx.NetworkError as e:
                raise LLM4AgentsError(str(e), "network_error", None, None) from e

            if res.status_code >= 400:
                body_text = res.text or ""
                if res.status_code == 402:
                    self._maybe_throw_x402(res, body_text)
                try:
                    err_body = res.json()
                except ValueError:
                    err_body = {}
                raise map_http_error(res.status_code, err_body, self._request_id(res))
            return res.json()

    async def post_with_meta(
        self, path: str, body: dict[str, Any]
    ) -> tuple[Any, httpx.Headers]:
        """Same as post() but returns (data, response_headers) together."""
        auth_headers = await self._build_post_headers(path, body)
        async with self._client(with_auth=False) as client:
            try:
                res = await client.post(
                    path, content=json.dumps(body), headers=auth_headers
                )
            except httpx.TimeoutException as e:
                raise LLM4AgentsError(str(e), "timeout", None, None) from e
            except httpx.NetworkError as e:
                raise LLM4AgentsError(str(e), "network_error", None, None) from e

            if res.status_code >= 400:
                body_text = res.text or ""
                if res.status_code == 402:
                    self._maybe_throw_x402(res, body_text)
                try:
                    err_body = res.json()
                except ValueError:
                    err_body = {}
                raise map_http_error(res.status_code, err_body, self._request_id(res))
            return res.json(), res.headers

    async def post_stream(self, path: str, body: dict[str, Any]) -> AsyncIterator[Any]:
        return self._stream(path, body)

    async def post_stream_with_meta(
        self, path: str, body: dict[str, Any]
    ) -> tuple[httpx.Headers, AsyncIterator[Any]]:
        """Like post_stream() but also returns the response headers captured at connection time.

        Returns ``(headers, async_iterator)`` so callers can read headers before
        iterating over SSE chunks.
        """
        auth_headers = await self._build_post_headers(path, body)
        client = self._client(with_auth=False)
        try:
            ctx = client.stream(
                "POST", path, content=json.dumps(body), headers=auth_headers
            )
            res = await ctx.__aenter__()
        except httpx.TimeoutException as e:
            await client.aclose()
            raise LLM4AgentsError(str(e), "timeout", None, None) from e
        except httpx.NetworkError as e:
            await client.aclose()
            raise LLM4AgentsError(str(e), "network_error", None, None) from e

        if res.status_code >= 400:
            raw = await res.aread()
            await ctx.__aexit__(None, None, None)
            await client.aclose()
            body_text = raw.decode("utf-8", errors="replace") if raw else ""
            if res.status_code == 402:
                self._maybe_throw_x402(res, body_text)
            try:
                err_body = json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                err_body = {}
            raise map_http_error(res.status_code, err_body, self._request_id(res))

        headers = res.headers

        async def _gen() -> AsyncIterator[Any]:
            try:
                async for chunk in self._parse_sse(res):
                    yield chunk
            finally:
                await ctx.__aexit__(None, None, None)
                await client.aclose()

        return headers, _gen()

    async def _stream(self, path: str, body: dict[str, Any]) -> AsyncIterator[Any]:
        auth_headers = await self._build_post_headers(path, body)
        async with self._client(with_auth=False) as client:
            try:
                async with client.stream(
                    "POST", path, content=json.dumps(body), headers=auth_headers
                ) as res:
                    if res.status_code >= 400:
                        raw = await res.aread()
                        body_text = raw.decode("utf-8", errors="replace") if raw else ""
                        if res.status_code == 402:
                            self._maybe_throw_x402(res, body_text)
                        try:
                            err_body = json.loads(raw)
                        except (ValueError, json.JSONDecodeError):
                            err_body = {}
                        raise map_http_error(
                            res.status_code, err_body, self._request_id(res)
                        )
                    async for chunk in self._parse_sse(res):
                        yield chunk
            except httpx.TimeoutException as e:
                raise LLM4AgentsError(str(e), "timeout", None, None) from e
            except httpx.NetworkError as e:
                raise LLM4AgentsError(str(e), "network_error", None, None) from e

    async def _parse_sse(self, res: httpx.Response) -> AsyncIterator[Any]:
        """Parse SSE chunks.

        Yields parsed JSON ``data:`` chunks AND ``{"_event": "<name>",
        "data": <json>}`` markers when an ``event:`` line precedes the
        ``data:`` line. This lets the chat layer dispatch the trailing
        ``event: x402-receipt`` chunk to its callback without having to
        rebuild the SSE state machine.

        Stops yielding chat chunks at ``data: [DONE]`` but KEEPS reading
        the stream so trailing event-typed chunks (x402-receipt) still
        surface.
        """
        current_event: str | None = None
        async for line in res.aiter_lines():
            stripped = line.strip()
            if stripped == "":
                # SSE event boundary
                current_event = None
                continue
            if stripped.startswith("event:"):
                current_event = stripped[len("event:"):].strip()
                continue
            if not stripped.startswith("data:"):
                continue
            payload = stripped[len("data:"):].strip()
            if current_event is not None:
                # Trailing typed event (e.g. x402-receipt). Surface to
                # the caller as a tagged dict; consumer decides what
                # to do.
                try:
                    parsed_event = json.loads(payload)
                except json.JSONDecodeError:
                    current_event = None
                    continue
                yield {"_event": current_event, "data": parsed_event}
                current_event = None
                continue
            if payload == "[DONE]":
                # Keep reading; an x402-receipt event may still arrive
                # AFTER [DONE] (proxy emits it post-settlement).
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue
