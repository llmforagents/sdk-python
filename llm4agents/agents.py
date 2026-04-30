from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from llm4agents.transport.http import HttpTransport


@dataclass(frozen=True)
class AgentRegistration:
    name: str
    uuid: str
    api_key: str
    request_id: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentRegistration:
        return cls(
            name=d.get("name", ""),
            uuid=d["uuid"],
            api_key=d["apiKey"],
            request_id=d.get("requestId"),
        )


@dataclass(frozen=True)
class AgentRegistrationParams:
    name: str


class Agents:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def register(self, name: str) -> AgentRegistration:
        data = await self._http.post("/api/v1/agents/register", {"name": name})
        return AgentRegistration.from_dict(data)
