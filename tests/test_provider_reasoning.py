"""Reasoning/chain-of-thought capture and request-side enabling per provider."""
from __future__ import annotations

from pathlib import Path
from decimal import Decimal

from solvent.cli.main import run_episode
from solvent.env.models import EnvConfig
from solvent.env.pricing import TokenUsage
from solvent.harness.llm import LLMHarness
from solvent.harness.model_client import FakeClient, ModelResponse, _coerce_response, response_to_dict
from solvent.harness.providers.anthropic import _anthropic_payload, _anthropic_reasoning
from solvent.harness.providers.base import ModelRequest
from solvent.harness.providers.google import google_generate_payload, google_response_to_model_response
from solvent.harness.providers.openai_compat import (
    OPENAI_COMPAT_PROVIDERS,
    OpenAICompatClient,
    openai_response_to_model_response,
)


def _request(model: str, reasoning: bool) -> ModelRequest:
    return ModelRequest(
        model=model,
        system_prompt="system",
        observation={"tick": 0},
        tools={"end_tick": {"description": "Advance", "input_schema": {"type": "object", "properties": {}}}},
        history=[],
        temperature=0.0,
        reasoning=reasoning,
    )


def test_openai_compat_captures_reasoning_content_and_normalized_reasoning() -> None:
    deepseek_style = openai_response_to_model_response(
        {"choices": [{"message": {"content": "answer", "reasoning_content": "step by step"}}]}
    )
    assert deepseek_style.reasoning == "step by step"
    assert deepseek_style.content == "answer"

    openrouter_style = openai_response_to_model_response(
        {"choices": [{"message": {"content": "answer", "reasoning": "normalized cot"}}]}
    )
    assert openrouter_style.reasoning == "normalized cot"

    block_style = openai_response_to_model_response(
        {"choices": [{"message": {"content": "", "reasoning": [{"text": "a"}, {"text": "b"}]}}]}
    )
    assert block_style.reasoning == "a\nb"


def test_openai_compat_requests_reasoning_only_on_openrouter() -> None:
    direct = OpenAICompatClient("minimax-m3", OPENAI_COMPAT_PROVIDERS["minimax"], api_key="x")
    assert direct._reasoning_is_requestable() is False

    routed = OpenAICompatClient(
        "minimax-m3", OPENAI_COMPAT_PROVIDERS["minimax"], api_key="x", base_url="https://openrouter.ai/api/v1"
    )
    assert routed._reasoning_is_requestable() is True


def test_anthropic_payload_enables_thinking_and_captures_blocks() -> None:
    payload = _anthropic_payload(_request("claude-opus-4-8", reasoning=True), max_tokens=128)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert payload["max_tokens"] == 128 + 1024  # headroom above the thinking budget
    assert "temperature" not in payload  # thinking requires the default temperature

    off = _anthropic_payload(_request("claude-opus-4-8", reasoning=False), max_tokens=128)
    assert "thinking" not in off

    reasoning = _anthropic_reasoning(
        {"content": [{"type": "thinking", "thinking": "deliberating"}, {"type": "text", "text": "answer"}]}
    )
    assert reasoning == "deliberating"


def test_google_payload_requests_thoughts_and_parse_splits_them() -> None:
    payload = google_generate_payload(_request("gemini-2.5-pro", reasoning=True), max_tokens=128)
    assert payload["generationConfig"]["thinkingConfig"] == {"includeThoughts": True}
    assert "thinkingConfig" not in google_generate_payload(_request("gemini-2.5-pro", reasoning=False), max_tokens=128)["generationConfig"]

    response = google_response_to_model_response(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking summary", "thought": True},
                            {"text": "visible answer"},
                            {"functionCall": {"name": "bid", "args": {"price": "1.00"}}},
                        ]
                    }
                }
            ]
        }
    )
    assert response.reasoning == "thinking summary"
    assert response.content == "visible answer"
    assert response.tool_call["name"] == "bid"


def test_reasoning_round_trips_through_sidecar_serialization() -> None:
    record = response_to_dict(ModelResponse({"name": "end_tick", "arguments": {}}, reasoning="cot"))
    assert record["reasoning"] == "cot"
    assert _coerce_response(record).reasoning == "cot"


def test_harness_requests_reasoning_and_records_it(tmp_path: Path) -> None:
    client = FakeClient([ModelResponse({"name": "end_tick", "arguments": {}}, TokenUsage(5, 1), "ok", "my reasoning")])
    trace_path = tmp_path / "reason.jsonl"
    run_episode(
        EnvConfig(
            seed=42,
            config_id="claude-opus-4-8:base",
            start_balance=Decimal("20.00"),
            horizon_ticks=1,
            overhead_per_tick=Decimal("0.05"),
            tool_call_cost=Decimal("0"),
            trace_path=trace_path,
            delivery_mode="tool_mediated",
        ),
        LLMHarness(model="claude-opus-4-8", client=client, max_turns=1),
    )

    assert client.requests[0].reasoning is True
    import json

    sidecar_row = json.loads(trace_path.with_suffix(".llm.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert sidecar_row["response"]["reasoning"] == "my reasoning"
    assert sidecar_row["response"]["content"] == "ok"
