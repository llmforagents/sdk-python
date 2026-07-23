from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoJobAccepted:
    id: str
    status: str
    polling_url: str
    charged_usd_cents: int


@dataclass(frozen=True)
class VideoJobStatus:
    id: str
    status: str
    charged_usd_cents: int
    video_url: str | None
    error: str | None
    refunded: bool | None


@dataclass(frozen=True)
class VideoContentResult:
    data: bytes
    content_type: str
    request_id: str | None
