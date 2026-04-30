from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Any
import json
import httpx
from llm4agents.errors import LLM4AgentsError, map_http_error


class HttpTransport:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
            http2=True,
        )

    def _request_id(self, response: httpx.Response) -> str | None:
        return response.headers.get("x-request-id")

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
        async with self._client() as client:
            try:
                res = await client.post(path, content=json.dumps(body))
            except httpx.TimeoutException as e:
                raise LLM4AgentsError(str(e), "timeout", None, None) from e
            except httpx.NetworkError as e:
                raise LLM4AgentsError(str(e), "network_error", None, None) from e
            if res.status_code >= 400:
                raise map_http_error(res.status_code, res.json(), self._request_id(res))
            return res.json()

    async def post_with_meta(self, path: str, body: dict[str, Any]) -> tuple[Any, httpx.Headers]:
        """Same as post() but returns (data, response_headers) together."""
        async with self._client() as client:
            try:
                res = await client.post(path, content=json.dumps(body))
            except httpx.TimeoutException as e:
                raise LLM4AgentsError(str(e), "timeout", None, None) from e
            except httpx.NetworkError as e:
                raise LLM4AgentsError(str(e), "network_error", None, None) from e
            if res.status_code >= 400:
                raise map_http_error(res.status_code, res.json(), self._request_id(res))
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
        # We need to hold the client open for the lifetime of the returned iterator.
        # We open the streaming context here, capture headers, and return a generator
        # that owns the client context manager.
        client = self._client()
        try:
            ctx = client.stream("POST", path, content=json.dumps(body))
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
            try:
                err_body = json.loads(raw)
            except Exception:
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
        async with self._client() as client:
            try:
                async with client.stream("POST", path, content=json.dumps(body)) as res:
                    if res.status_code >= 400:
                        raw = await res.aread()
                        try:
                            err_body = json.loads(raw)
                        except Exception:
                            err_body = {}
                        raise map_http_error(res.status_code, err_body, self._request_id(res))
                    async for chunk in self._parse_sse(res):
                        yield chunk
            except httpx.TimeoutException as e:
                raise LLM4AgentsError(str(e), "timeout", None, None) from e
            except httpx.NetworkError as e:
                raise LLM4AgentsError(str(e), "network_error", None, None) from e

    async def _parse_sse(self, res: httpx.Response) -> AsyncIterator[Any]:
        async for line in res.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue
