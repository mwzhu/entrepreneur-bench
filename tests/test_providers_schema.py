import json

import pytest

from solvent.env.pricing import TokenUsage
from solvent.harness.model_client import ModelRequest, client_for_model
from solvent.harness.providers.google import google_generate_payload, google_response_to_model_response
from solvent.harness.providers.openai_compat import (
    OPENROUTER_BASE_URL,
    OPENROUTER_KEY_URL,
    OpenAICompatClient,
    openai_chat_payload,
    openai_response_to_model_response,
    probe_openrouter_chat,
    probe_openrouter_key,
    provider_for_model,
)
from solvent.harness.providers.schema_translate import to_google_function_declarations, to_openai_tools


def test_neutral_tool_schema_translates_to_openai_and_google_shapes() -> None:
    tools = {
        "bid": {
            "description": "Submit bid",
            "input_schema": {
                "type": "object",
                "properties": {"price": {"type": "string"}},
                "required": ["price"],
                "additionalProperties": False,
            },
        }
    }

    openai = to_openai_tools(tools)
    google = to_google_function_declarations(tools)

    assert openai[0]["type"] == "function"
    assert openai[0]["function"]["name"] == "bid"
    assert openai[0]["function"]["parameters"]["required"] == ["price"]
    assert openai[0]["function"]["parameters"]["additionalProperties"] is False
    assert google[0]["name"] == "bid"
    assert google[0]["parameters"]["properties"]["price"]["type"] == "string"
    assert "additionalProperties" not in google[0]["parameters"]


def test_openai_compatible_payload_and_response_normalize_usage() -> None:
    request = _request("gpt-test")

    payload = openai_chat_payload(request, max_tokens=64)
    response = openai_response_to_model_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "bid",
                                    "arguments": json.dumps({"job_id": "job-1", "price": "1.00"}),
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 25},
            },
        }
    )

    assert payload["model"] == "gpt-test"
    assert payload["max_completion_tokens"] == 64
    assert "max_tokens" not in payload
    assert payload["tools"][0]["function"]["name"] == "bid"
    assert response.tool_call == {"name": "bid", "arguments": {"job_id": "job-1", "price": "1.00"}}
    assert response.usage == TokenUsage(input_tokens=75, output_tokens=10, cache_read_tokens=25)


def test_openai_compatible_non_gpt_payload_uses_legacy_token_limit_field() -> None:
    payload = openai_chat_payload(_request("kimi-k2-6"), max_tokens=64)

    assert payload["max_tokens"] == 64
    assert "max_completion_tokens" not in payload


def test_google_payload_and_response_normalize_usage() -> None:
    request = _request("gemini-test")

    payload = google_generate_payload(request, max_tokens=64)
    response = google_response_to_model_response(
        {
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "end_tick", "args": {}}}]}}
            ],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 5,
                "cachedContentTokenCount": 20,
            },
        }
    )

    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "bid"
    assert payload["systemInstruction"]["parts"][0]["text"] == "system"
    assert response.tool_call == {"name": "end_tick", "arguments": {}}
    assert response.usage == TokenUsage(input_tokens=30, output_tokens=5, cache_read_tokens=20)


def test_client_factory_routes_new_provider_families_and_fails_without_keys(monkeypatch) -> None:
    for key in ["OPENAI_API_KEY", "MOONSHOT_API_KEY", "ZHIPU_API_KEY", "MINIMAX_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    assert provider_for_model("gpt-5-mini").api_key_env == "OPENAI_API_KEY"
    assert provider_for_model("kimi-k2-6").api_key_env == "MOONSHOT_API_KEY"
    assert provider_for_model("glm-latest").api_key_env == "ZHIPU_API_KEY"
    assert provider_for_model("minimax-latest").api_key_env == "MINIMAX_API_KEY"
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        client_for_model("gpt-5-mini")
    with pytest.raises(RuntimeError, match="MOONSHOT_API_KEY"):
        client_for_model("kimi-k2-6")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY|GEMINI_API_KEY"):
        client_for_model("gemini-pro")


def test_openrouter_key_can_back_openai_compatible_provider_families(monkeypatch) -> None:
    for key in ["MOONSHOT_API_KEY", "ZHIPU_API_KEY", "MINIMAX_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")

    kimi = client_for_model("kimi-k2.6")
    glm = client_for_model("glm-5")
    minimax = client_for_model("minimax-m3")
    request = _request("kimi-k2.6")

    assert isinstance(kimi, OpenAICompatClient)
    assert isinstance(glm, OpenAICompatClient)
    assert isinstance(minimax, OpenAICompatClient)
    assert kimi.base_url == OPENROUTER_BASE_URL
    assert kimi._model_for_request(request) == "moonshotai/kimi-k2.6"
    assert glm._model_for_request(_request("glm-5")) == "z-ai/glm-5"
    assert minimax._model_for_request(_request("minimax-m3")) == "minimax/minimax-m3"


def test_openrouter_fallback_respects_model_alias(monkeypatch) -> None:
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("SOLVENT_MODEL_ALIAS_KIMI_K2_6", "moonshotai/kimi-k2.6:free")
    client = client_for_model("kimi-k2.6")

    assert isinstance(client, OpenAICompatClient)
    assert client._model_for_request(_request("kimi-k2.6")) == "moonshotai/kimi-k2.6:free"


def test_probe_openrouter_chat_uses_gateway_model_alias_and_reports_failure(monkeypatch) -> None:
    calls = []

    def fake_urlopen_json(request, **kwargs):
        calls.append((request, kwargs))
        if request.full_url == OPENROUTER_KEY_URL:
            return {"data": {"is_management_key": False}}
        raise RuntimeError('provider API error 401: {"error":{"message":"User not found."}}')

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("SOLVENT_MODEL_ALIAS_KIMI_K2_6", "moonshotai/kimi-k2.6:free")
    monkeypatch.setattr("solvent.harness.providers.openai_compat.urlopen_json", fake_urlopen_json)

    ok, detail = probe_openrouter_chat("kimi-k2.6")
    body = calls[1][0].data.decode("utf-8")

    assert ok is False
    assert "User not found" in detail
    assert "openrouter-secret" not in detail
    assert '"model": "moonshotai/kimi-k2.6:free"' in body
    assert calls[0][1]["max_attempts"] == 1
    assert calls[1][1]["max_attempts"] == 1


def test_probe_openrouter_key_rejects_management_keys(monkeypatch) -> None:
    def fake_urlopen_json(request, **kwargs):
        return {"data": {"is_management_key": True, "label": "sk-or-v1-..."}}

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setattr("solvent.harness.providers.openai_compat.urlopen_json", fake_urlopen_json)

    ok, detail = probe_openrouter_key()

    assert ok is False
    assert "management key" in detail
    assert "openrouter-secret" not in detail


def _request(model: str) -> ModelRequest:
    return ModelRequest(
        model=model,
        system_prompt="system",
        observation={"tick": 0},
        tools={
            "bid": {
                "description": "Submit bid",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "price": {"type": "string"},
                    },
                    "required": ["job_id", "price"],
                    "additionalProperties": False,
                },
            }
        },
        history=[{"observation": {"tick": 0}}],
        temperature=0.2,
    )
