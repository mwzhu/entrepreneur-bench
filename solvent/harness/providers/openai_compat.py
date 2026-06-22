from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

from solvent.env.pricing import TokenUsage
from solvent.harness.providers.base import ModelRequest, ModelResponse, resolve_model_name
from solvent.harness.providers.http import urlopen_json
from solvent.harness.providers.schema_translate import to_openai_tools


@dataclass(frozen=True)
class OpenAICompatProvider:
    base_url: str
    api_key_env: str
    openrouter_model: str | None = None


OPENAI_COMPAT_PROVIDERS: dict[str, OpenAICompatProvider] = {
    "gpt": OpenAICompatProvider("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "kimi": OpenAICompatProvider("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "moonshotai/kimi-k2.6"),
    "moonshot": OpenAICompatProvider("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "moonshotai/kimi-k2.6"),
    "glm": OpenAICompatProvider("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", "z-ai/glm-5"),
    "minimax": OpenAICompatProvider("https://api.minimax.io/v1", "MINIMAX_API_KEY", "minimax/minimax-m3"),
}

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_URL = f"{OPENROUTER_BASE_URL}/key"
OPENROUTER_REFERER = "https://localhost"
OPENROUTER_TITLE = "Solvent"


class OpenAICompatClient:
    """OpenAI-compatible Chat Completions client for GPT and compatible providers."""

    def __init__(
        self,
        model: str,
        provider: OpenAICompatProvider,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.model = model
        self.provider = provider
        self.api_key = api_key or os.environ.get(provider.api_key_env)
        self.base_url = (base_url or os.environ.get(_base_url_env(provider.api_key_env)) or provider.base_url).rstrip("/")
        self.openrouter_fallback = False
        self.max_tokens = max_tokens
        self.temperature = temperature
        if not self.api_key:
            openrouter_key = os.environ.get(OPENROUTER_API_KEY_ENV)
            if openrouter_key and provider.openrouter_model:
                self.api_key = openrouter_key
                self.base_url = OPENROUTER_BASE_URL
                self.openrouter_fallback = True
            else:
                raise RuntimeError(_missing_key_message(provider))

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = openai_chat_payload(
            request,
            self.max_tokens,
            self.temperature,
            model_override=self._model_for_request(request),
        )
        if request.reasoning and self._reasoning_is_requestable():
            # OpenRouter normalizes reasoning across providers behind this flag;
            # direct reasoning models (e.g. MiniMax) return reasoning_content by default.
            payload["reasoning"] = {"enabled": True}
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        data = urlopen_json(http_request)
        return openai_response_to_model_response(data)

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", "authorization": f"Bearer {self.api_key}"}
        if self.openrouter_fallback:
            headers["HTTP-Referer"] = OPENROUTER_REFERER
            headers["X-OpenRouter-Title"] = OPENROUTER_TITLE
        return headers

    def _model_for_request(self, request: ModelRequest) -> str | None:
        if not self.openrouter_fallback:
            return None
        resolved = resolve_model_name(request.model)
        return resolved if resolved != request.model else self.provider.openrouter_model

    def _reasoning_is_requestable(self) -> bool:
        # Only attach the reasoning flag where it is a documented request field;
        # bare OpenAI Chat Completions rejects unknown keys.
        return self.openrouter_fallback or "openrouter.ai" in self.base_url


def provider_for_model(model: str) -> OpenAICompatProvider | None:
    family = model.split("-", 1)[0]
    if model.startswith("gpt-"):
        family = "gpt"
    return OPENAI_COMPAT_PROVIDERS.get(family)


def openai_chat_payload(
    request: ModelRequest,
    max_tokens: int,
    temperature: float | None = None,
    model_override: str | None = None,
) -> dict[str, Any]:
    messages = [{"role": "system", "content": request.system_prompt}]
    for item in request.history:
        messages.append({"role": "user", "content": json.dumps(item, sort_keys=True)})
    messages.append({"role": "user", "content": json.dumps({"observation": request.observation}, sort_keys=True)})
    model = model_override or resolve_model_name(request.model)
    payload = {
        "model": model,
        "messages": messages,
        "tools": to_openai_tools(request.tools),
        "tool_choice": "auto",
        "temperature": request.temperature if temperature is None else temperature,
    }
    if model.startswith("gpt-"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    return payload


def openai_response_to_model_response(data: dict[str, Any]) -> ModelResponse:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    tool_call = _extract_tool_call(message)
    usage = data.get("usage", {})
    prompt_details = usage.get("prompt_tokens_details", {})
    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    cache_read_tokens = int(prompt_details.get("cached_tokens", usage.get("cached_tokens", 0)))
    return ModelResponse(
        tool_call=tool_call,
        usage=TokenUsage(
            input_tokens=max(0, prompt_tokens - cache_read_tokens),
            output_tokens=int(usage.get("completion_tokens", usage.get("output_tokens", 0))),
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=int(usage.get("cache_creation_input_tokens", usage.get("cache_write_tokens", 0))),
        ),
        content=str(message.get("content") or ""),
        reasoning=_extract_openai_reasoning(message),
    )


def _extract_openai_reasoning(message: dict[str, Any]) -> str:
    # reasoning_content: DeepSeek / MiniMax / GLM / Kimi. reasoning: OpenRouter-normalized.
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if isinstance(reasoning, list):  # some providers return a list of reasoning blocks
        reasoning = "\n".join(
            str(block.get("text", block) if isinstance(block, dict) else block) for block in reasoning
        )
    return str(reasoning or "")


def _extract_tool_call(message: dict[str, Any]) -> dict[str, Any]:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        function = tool_calls[0].get("function", {})
        return {"name": function.get("name", "end_tick"), "arguments": _json_object(function.get("arguments", {}))}
    content = message.get("content") or ""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "name" in parsed:
            return {"name": parsed["name"], "arguments": parsed.get("arguments", {})}
    except json.JSONDecodeError:
        pass
    return {"name": "end_tick", "arguments": {}}


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _base_url_env(api_key_env: str) -> str:
    return api_key_env.replace("_API_KEY", "_BASE_URL")


def _missing_key_message(provider: OpenAICompatProvider) -> str:
    if provider.openrouter_model:
        return f"{provider.api_key_env} or {OPENROUTER_API_KEY_ENV} is required for live runs"
    return f"{provider.api_key_env} is required for live runs"


def probe_openrouter_chat(model: str, *, timeout: int = 30) -> tuple[bool, str]:
    provider = provider_for_model(model)
    if provider is None or not provider.openrouter_model:
        return True, "OpenRouter probe not applicable"
    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        return False, f"{OPENROUTER_API_KEY_ENV} missing"
    key_ok, key_detail = probe_openrouter_key(timeout=timeout)
    if not key_ok:
        return False, key_detail
    model_id = resolve_model_name(model)
    if model_id == model:
        model_id = provider.openrouter_model
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply ok."}],
        "max_tokens": 1,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-OpenRouter-Title": OPENROUTER_TITLE,
        },
        method="POST",
    )
    try:
        urlopen_json(request, timeout=timeout, max_attempts=1)
    except RuntimeError as exc:
        return False, _redact_provider_error(str(exc))
    return True, f"OpenRouter chat probe passed for {model_id}"


def probe_openrouter_key(*, timeout: int = 30) -> tuple[bool, str]:
    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        return False, f"{OPENROUTER_API_KEY_ENV} missing"
    request = urllib.request.Request(
        OPENROUTER_KEY_URL,
        headers={"authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        data = urlopen_json(request, timeout=timeout, max_attempts=1)
    except RuntimeError as exc:
        return False, _redact_provider_error(str(exc))
    key_data = data.get("data", {}) if isinstance(data, dict) else {}
    if key_data.get("is_management_key"):
        return False, f"{OPENROUTER_API_KEY_ENV} is a management key; create a standard OpenRouter API key for chat completions"
    return True, "OpenRouter API key is usable for inference"


def _redact_provider_error(error: str) -> str:
    return error[:240]
