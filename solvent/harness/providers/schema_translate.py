from __future__ import annotations

from typing import Any

GOOGLE_UNSUPPORTED_SCHEMA_KEYS = {"additionalProperties", "$schema"}


def to_openai_tools(tools: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": str(schema.get("description", f"Solvent environment tool: {name}")),
                "parameters": dict(schema.get("input_schema", {"type": "object", "properties": {}, "required": []})),
            },
        }
        for name, schema in tools.items()
    ]


def to_google_function_declarations(tools: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": str(schema.get("description", f"Solvent environment tool: {name}")),
            "parameters": _google_schema(schema.get("input_schema", {"type": "object", "properties": {}, "required": []})),
        }
        for name, schema in tools.items()
    ]


def _google_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _google_schema(child)
            for key, child in value.items()
            if key not in GOOGLE_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [_google_schema(item) for item in value]
    return value
