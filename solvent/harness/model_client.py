from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from solvent.env.pricing import TokenUsage
from solvent.harness.providers.anthropic import AnthropicMessagesClient, _anthropic_payload, _anthropic_tool
from solvent.harness.providers.base import ModelClient, ModelRequest, ModelResponse, model_alias_env_var, resolve_model_name
from solvent.harness.providers.google import GoogleGenerativeClient
from solvent.harness.providers.openai_compat import OpenAICompatClient, provider_for_model


class FakeClient:
    """Deterministic client for tests and local harness development."""

    def __init__(self, responses: list[ModelResponse | dict[str, Any]]):
        self._responses = [_coerce_response(response) for response in responses]
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            return ModelResponse({"name": "end_tick", "arguments": {}}, usage=TokenUsage(input_tokens=1, output_tokens=1))
        return self._responses.pop(0)


class RecordedClient:
    """Replay a previously recorded LLM sidecar without making API calls."""

    def __init__(self, path: Path):
        self._responses = [
            _coerce_response(json.loads(line)["response"])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("recorded model sidecar is exhausted")
        return self._responses.pop(0)


def client_for_model(model: str, temperature: float = 0.0, max_tokens: int = 1024) -> ModelClient:
    if model == "fake":
        return FakeClient([])
    if model == "recorded":
        raise RuntimeError("recorded runs require an explicit RecordedClient(path)")
    if model.startswith("claude-"):
        return AnthropicMessagesClient(temperature=temperature, max_tokens=max_tokens)
    if model.startswith("gemini-"):
        return GoogleGenerativeClient(model, temperature=temperature, max_tokens=max_tokens)
    provider = provider_for_model(model)
    if provider is not None:
        return OpenAICompatClient(model, provider, temperature=temperature, max_tokens=max_tokens)
    raise RuntimeError(f"no live model client configured for model: {model}")


def response_to_dict(response: ModelResponse) -> dict[str, Any]:
    return {
        "tool_call": response.tool_call,
        "usage": asdict(response.usage),
        "content": response.content,
        "reasoning": response.reasoning,
    }


def _coerce_response(response: ModelResponse | dict[str, Any]) -> ModelResponse:
    if isinstance(response, ModelResponse):
        return response
    usage_raw = response.get("usage", {})
    return ModelResponse(
        tool_call=dict(response.get("tool_call", response)),
        usage=TokenUsage(
            input_tokens=int(usage_raw.get("input_tokens", 0)),
            output_tokens=int(usage_raw.get("output_tokens", 0)),
            cache_read_tokens=int(usage_raw.get("cache_read_tokens", usage_raw.get("cache_read_input_tokens", 0))),
            cache_write_tokens=int(usage_raw.get("cache_write_tokens", usage_raw.get("cache_creation_input_tokens", 0))),
        ),
        content=str(response.get("content", "")),
        reasoning=str(response.get("reasoning", "")),
    )

