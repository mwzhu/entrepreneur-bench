from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from solvent.env.pricing import TokenUsage
from solvent.harness.providers.base import ModelRequest, ModelResponse, resolve_model_name
from solvent.harness.providers.http import urlopen_json

# Extended-thinking budget (billed as output tokens) when reasoning is requested.
REASONING_BUDGET_TOKENS = 1024


class AnthropicMessagesClient:
    """Minimal stdlib Anthropic Messages client with tool-call support."""

    endpoint = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str | None = None, max_tokens: int = 1024, temperature: float = 0.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for live Anthropic model runs")

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = _anthropic_payload(request, self.max_tokens)
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        data = urlopen_json(http_request)

        tool_call = _extract_anthropic_tool_call(data)
        usage = data.get("usage", {})
        return ModelResponse(
            tool_call=tool_call,
            usage=TokenUsage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                cache_read_tokens=int(usage.get("cache_read_input_tokens", usage.get("cache_read_tokens", 0))),
                cache_write_tokens=int(usage.get("cache_creation_input_tokens", usage.get("cache_write_tokens", 0))),
            ),
            content=_anthropic_text(data),
            reasoning=_anthropic_reasoning(data),
        )


def _anthropic_tool(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": str(schema.get("description", f"Solvent environment tool: {name}")),
        "input_schema": dict(schema.get("input_schema", {"type": "object", "properties": {}, "required": []})),
    }


def _anthropic_payload(request: ModelRequest, max_tokens: int) -> dict[str, Any]:
    model = resolve_model_name(request.model)
    system_block = {
        "type": "text",
        "text": request.system_prompt,
    }
    if request.cache_hint:
        system_block["cache_control"] = {"type": "ephemeral"}
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [system_block],
        "messages": _anthropic_messages(request),
        "tools": [_anthropic_tool(name, schema) for name, schema in request.tools.items()],
        "tool_choice": {"type": "auto"},
    }
    if request.reasoning and _anthropic_supports_thinking(model):
        # Extended thinking requires headroom above the budget and the default
        # temperature; the thinking tokens are billed as output.
        payload["thinking"] = {"type": "enabled", "budget_tokens": REASONING_BUDGET_TOKENS}
        payload["max_tokens"] = max_tokens + REASONING_BUDGET_TOKENS
    elif _anthropic_supports_sampling_params(model):
        payload["temperature"] = request.temperature
    return payload


def _anthropic_supports_sampling_params(model: str) -> bool:
    return not model.startswith("claude-opus-4-8")


def _anthropic_supports_thinking(model: str) -> bool:
    return model.startswith(("claude-opus-4", "claude-sonnet-4", "claude-haiku-4", "claude-3-7-sonnet"))


def _anthropic_messages(request: ModelRequest) -> list[dict[str, Any]]:
    messages = []
    for index, item in enumerate(request.history):
        block: dict[str, Any] = {"type": "text", "text": json.dumps(item, sort_keys=True)}
        if request.cache_hint and index == len(request.history) - 1:
            block["cache_control"] = {"type": "ephemeral"}
        messages.append({"role": "user", "content": [block]})
    messages.append({"role": "user", "content": json.dumps({"observation": request.observation}, sort_keys=True)})
    return messages


def _extract_anthropic_tool_call(data: dict[str, Any]) -> dict[str, Any]:
    for block in data.get("content", []):
        if block.get("type") == "tool_use":
            return {"name": block["name"], "arguments": block.get("input", {})}
    text = _anthropic_text(data)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "name" in parsed:
            return {"name": parsed["name"], "arguments": parsed.get("arguments", {})}
    except json.JSONDecodeError:
        pass
    return {"name": "end_tick", "arguments": {}}


def _anthropic_text(data: dict[str, Any]) -> str:
    chunks = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    return "\n".join(chunk for chunk in chunks if chunk)


def _anthropic_reasoning(data: dict[str, Any]) -> str:
    chunks = []
    for block in data.get("content", []):
        if block.get("type") == "thinking":
            chunks.append(str(block.get("thinking", "")))
        elif block.get("type") == "redacted_thinking":
            chunks.append("[redacted thinking]")
    return "\n".join(chunk for chunk in chunks if chunk)
