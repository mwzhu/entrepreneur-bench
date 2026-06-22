from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from solvent.env.pricing import TokenUsage
from solvent.harness.providers.base import ModelRequest, ModelResponse, resolve_model_name
from solvent.harness.providers.http import urlopen_json
from solvent.harness.providers.schema_translate import to_google_function_declarations


class GoogleGenerativeClient:
    """Minimal Gemini native client with function-call support."""

    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024, temperature: float = 0.0):
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required for live Gemini runs")

    def complete(self, request: ModelRequest) -> ModelResponse:
        model = resolve_model_name(request.model)
        url = self.endpoint.format(model=model)
        payload = google_generate_payload(request, self.max_tokens)
        http_request = urllib.request.Request(
            f"{url}?key={self.api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        data = urlopen_json(http_request)
        return google_response_to_model_response(data)


def google_generate_payload(request: ModelRequest, max_tokens: int) -> dict[str, Any]:
    parts = [
        {"text": json.dumps(item, sort_keys=True)}
        for item in request.history
    ]
    parts.append({"text": json.dumps({"observation": request.observation}, sort_keys=True)})
    generation_config: dict[str, Any] = {"maxOutputTokens": max_tokens, "temperature": request.temperature}
    if request.reasoning:
        # Gemini 2.5 thinking models return thought summaries only when asked.
        generation_config["thinkingConfig"] = {"includeThoughts": True}
    return {
        "systemInstruction": {"parts": [{"text": request.system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
        "tools": [{"functionDeclarations": to_google_function_declarations(request.tools)}],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": generation_config,
    }


def google_response_to_model_response(data: dict[str, Any]) -> ModelResponse:
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    tool_call = {"name": "end_tick", "arguments": {}}
    text_chunks = []
    reasoning_chunks = []
    for part in parts:
        if "functionCall" in part:
            call = part["functionCall"]
            tool_call = {"name": call.get("name", "end_tick"), "arguments": dict(call.get("args", {}))}
        if "text" in part:
            # A part flagged thought=True is a reasoning summary, not the answer.
            if part.get("thought"):
                reasoning_chunks.append(str(part["text"]))
            else:
                text_chunks.append(str(part["text"]))
    usage = data.get("usageMetadata", {})
    prompt_tokens = int(usage.get("promptTokenCount", 0))
    cache_read_tokens = int(usage.get("cachedContentTokenCount", 0))
    return ModelResponse(
        tool_call=tool_call,
        usage=TokenUsage(
            input_tokens=max(0, prompt_tokens - cache_read_tokens),
            output_tokens=int(usage.get("candidatesTokenCount", 0)),
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=0,
        ),
        content="\n".join(text_chunks),
        reasoning="\n".join(reasoning_chunks),
    )
