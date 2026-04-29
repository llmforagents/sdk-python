from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TypedDict, Union


class ToolDefinition(TypedDict):
    name: str
    description: str
    inputSchema: dict[str, Any]


@dataclass(frozen=True)
class McpTextContent:
    type: str  # always "text"
    text: str


@dataclass(frozen=True)
class McpImageContent:
    type: str  # always "image"
    data: str
    mimeType: str


@dataclass(frozen=True)
class McpResourceContent:
    type: str  # always "resource"
    uri: str
    text: str | None = None
    mimeType: str | None = None


McpContent = Union[McpTextContent, McpImageContent, McpResourceContent]


@dataclass(frozen=True)
class McpToolResult:
    content: tuple[McpContent, ...]
    text: str

    def __str__(self) -> str:
        return self.text
