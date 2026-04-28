from typing import Any, TypedDict


class ToolDefinition(TypedDict):
    name: str
    description: str
    inputSchema: dict[str, Any]
