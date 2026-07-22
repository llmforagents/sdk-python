from __future__ import annotations
from llm4agents.transport.http import HttpTransport
from llm4agents.audio.types import SpeechResult


class Speech:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def create(
        self,
        *,
        model: str,
        input: str,
        voice: str,
        response_format: str | None = None,
        speed: float | None = None,
    ) -> SpeechResult:
        payload: dict[str, object] = {"model": model, "input": input, "voice": voice}
        if response_format is not None:
            payload["response_format"] = response_format
        if speed is not None:
            payload["speed"] = speed
        data, headers = await self._http.post_binary("/v1/audio/speech", payload)
        cents = headers.get("x-charged-usd-cents")
        return SpeechResult(
            data=data,
            content_type=headers.get("content-type") or "audio/mpeg",
            request_id=headers.get("x-request-id"),
            charged_usd_cents=int(cents) if cents is not None else None,
            model_used=headers.get("x-model-used"),
        )


class Audio:
    def __init__(self, http: HttpTransport) -> None:
        self.speech = Speech(http)
