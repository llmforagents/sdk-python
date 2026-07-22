from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechResult:
    data: bytes
    content_type: str
    request_id: str | None
    charged_usd_cents: int | None
    model_used: str | None
