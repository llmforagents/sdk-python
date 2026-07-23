from __future__ import annotations
from urllib.parse import quote
from llm4agents.transport.http import HttpTransport
from llm4agents.videos.types import VideoContentResult, VideoJobAccepted, VideoJobStatus


class Videos:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def create(
        self,
        *,
        prompt: str,
        model: str | None = None,
        image: str | None = None,
        duration: int | None = None,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        generate_audio: bool | None = None,
        seed: int | None = None,
    ) -> VideoJobAccepted:
        payload: dict[str, object] = {"prompt": prompt}
        for key, value in (
            ("model", model), ("image", image), ("duration", duration),
            ("resolution", resolution), ("aspect_ratio", aspect_ratio),
            ("generate_audio", generate_audio), ("seed", seed),
        ):
            if value is not None:
                payload[key] = value
        data = await self._http.post("/v1/videos", payload)
        return VideoJobAccepted(
            id=data["id"], status=data["status"],
            polling_url=data["polling_url"], charged_usd_cents=data["charged_usd_cents"],
        )

    async def get(self, job_id: str) -> VideoJobStatus:
        data = await self._http.get(f"/v1/videos/{quote(job_id, safe='')}")
        return VideoJobStatus(
            id=data["id"], status=data["status"],
            charged_usd_cents=data.get("charged_usd_cents", 0),
            video_url=data.get("video_url"), error=data.get("error"),
            refunded=data.get("refunded"),
        )

    async def content(self, job_id: str) -> VideoContentResult:
        data, headers = await self._http.get_binary(f"/v1/videos/{quote(job_id, safe='')}/content")
        return VideoContentResult(
            data=data,
            content_type=headers.get("content-type") or "video/mp4",
            request_id=headers.get("x-request-id"),
        )
