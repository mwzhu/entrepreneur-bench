import json
from decimal import Decimal
from pathlib import Path

import pytest

from solvent.cli.main import run_episode
from solvent.env.env import Environment
from solvent.env.models import EnvConfig
from solvent.env.pricing import TokenUsage
from solvent.harness.llm import BudgetExceededError, LLMHarness, parse_ablation_spec
from solvent.harness.model_client import (
    FakeClient,
    ModelRequest,
    ModelResponse,
    _anthropic_payload,
    _anthropic_tool,
    model_alias_env_var,
    resolve_model_name,
)
from solvent.scoring.scorecard import score_trace
from solvent.harness.prompts import system_prompt


def test_fake_client_drives_tool_mediated_episode_and_records_compute(tmp_path: Path) -> None:
    client = FakeClient(
        [
            _response("list_jobs", {}, 100, 10),
            _response("inspect_job", {"job_id": "dc-42-0"}, 80, 8),
            _response("bid", {"job_id": "dc-42-0", "price": "0.50"}, 70, 7),
            _response("list_models", {}, 60, 6),
            _response("deliver", {"job_id": "dc-42-0", "model": "tool-pro"}, 90, 9),
            _response("end_tick", {}, 40, 4),
        ]
    )
    trace_path = tmp_path / "llm.jsonl"
    summary = run_episode(
        _config(trace_path),
        LLMHarness(model="claude-opus-4-8", ablations={"procedure"}, client=client),
    )

    events = _events(trace_path)
    paid_revenue = next(Decimal(event["payload"]["revenue"]) for event in events if event["kind"] == "paid")
    # Brain spend is metered separately; business balance only sees delivery price,
    # overhead, and realized contract revenue.
    assert summary.end_balance == Decimal("1000.00") - Decimal("45.00") - Decimal("0.05") + paid_revenue
    sidecar = trace_path.with_suffix(".llm.jsonl")
    assert sidecar.exists()
    sidecar_rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
    assert len(sidecar_rows) == 6
    assert "Forced procedure" in sidecar_rows[0]["request"]["system_prompt"]
    assert "bid" in sidecar_rows[0]["request"]["tools"]
    assert "available_jobs" in sidecar_rows[0]["request"]["observation"]

    assert [event["kind"] for event in events].count("brain_metered") == 6
    assert "delivery_passed" in [event["kind"] for event in events]
    assert all(event["burn_delta"] == "0" for event in events if event["kind"] == "brain_metered")

    scorecard = score_trace(trace_path)
    assert scorecard.compute is not None
    assert scorecard.compute.brain_tokens_in == 440
    assert scorecard.compute.brain_tokens_out == 44
    assert scorecard.compute.brain_cost > Decimal("0")
    assert scorecard.delivery.pass_rate == 1.0
    assert scorecard.tool_selection is not None
    assert scorecard.tool_selection.tool_price_charged == Decimal("45.00")


def test_malformed_model_tool_call_becomes_invalid_action(tmp_path: Path) -> None:
    client = FakeClient([ModelResponse({"name": "not_a_tool", "arguments": {}}, usage=TokenUsage(1, 1))])
    summary = run_episode(
        _config(tmp_path / "invalid.jsonl"),
        LLMHarness(model="fake", client=client, max_turns=1),
    )

    invalid = [event for event in _events(summary.trace_path) if event["kind"] == "invalid_action"]
    assert len(invalid) == 1
    assert invalid[0]["payload"]["code"] == "malformed_tool_call"


def test_harness_raises_and_traces_budget_exceeded(tmp_path: Path) -> None:
    client = FakeClient([_response("end_tick", {}, 1000, 100)])
    trace_path = tmp_path / "budget.jsonl"
    env = Environment(_config(trace_path))
    try:
        with pytest.raises(BudgetExceededError):
            LLMHarness(
                model="claude-opus-4-8",
                client=client,
                max_turns=1,
                budget_limit=Decimal("0.000001"),
            ).run(env)
    finally:
        env.finalize()

    events = _events(trace_path)
    assert any(event["kind"] == "brain_metered" for event in events)
    exceeded = [event for event in events if event["kind"] == "budget_exceeded"]
    assert len(exceeded) == 1
    assert Decimal(exceeded[0]["payload"]["cumulative_cost"]) > Decimal(exceeded[0]["payload"]["budget_limit"])


def test_ablation_spec_is_order_insensitive_and_rejects_base_mix() -> None:
    assert parse_ablation_spec("+procedure+memory+economic") == {"memory", "procedure", "economic"}
    try:
        parse_ablation_spec("base+procedure")
    except ValueError as exc:
        assert "invalid ablation" in str(exc)
    else:
        raise AssertionError("base mixed with ablations should fail")


def test_system_prompt_base_states_mechanics_without_strategy_scaffolds() -> None:
    prompt = system_prompt(set())

    assert "Your goal is to maximize your final balance" in prompt
    assert "visible starting_price" in prompt
    assert "hidden client ceiling" in prompt
    assert "delivery model from the public menu" in prompt
    assert "single delivery attempt per job" in prompt
    assert "charge applies whether or not the delivery passes" in prompt
    assert "Calling end_tick advances business time" in prompt
    assert "Prefer actions with positive expected value" not in prompt
    assert "Counter just below" not in prompt
    assert "Do not advance time" not in prompt
    assert "price relative to the visible starting_price" not in prompt


def test_economic_ablation_adds_general_ev_guidance_without_ceiling_recipe() -> None:
    prompt = system_prompt({"economic"})

    assert "Prefer actions with positive expected value" in prompt
    assert "customer concessions" in prompt
    assert "Counter just below" not in prompt


def test_anthropic_tool_uses_adapter_input_schema() -> None:
    tool = _anthropic_tool(
        "bid",
        {
            "description": "Submit bid",
            "input_schema": {
                "type": "object",
                "properties": {"price": {"type": "string"}},
                "required": ["price"],
                "additionalProperties": False,
            },
        },
    )
    assert tool["description"] == "Submit bid"
    assert tool["input_schema"]["required"] == ["price"]
    assert tool["input_schema"]["properties"]["price"]["type"] == "string"


def test_anthropic_payload_omits_default_temperature_for_opus_4_8() -> None:
    payload = _anthropic_payload(_request("claude-opus-4-8", temperature=0), max_tokens=128)

    assert payload["model"] == "claude-opus-4-8"
    assert payload["max_tokens"] == 128
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in payload


def test_anthropic_payload_omits_cache_control_when_caching_disabled() -> None:
    payload = _anthropic_payload(_request("claude-opus-4-8", temperature=0, cache_hint=False), max_tokens=128)

    assert "cache_control" not in payload["system"][0]


def test_anthropic_payload_caches_stable_history_prefix_only() -> None:
    request = ModelRequest(
        model="claude-sonnet-4-6",
        system_prompt="system",
        observation={"tick": 1},
        tools={"end_tick": {"description": "Advance", "input_schema": {"type": "object", "properties": {}}}},
        history=[{"observation": {"tick": 0}, "tool_call": {"name": "list_jobs", "arguments": {}}}],
        temperature=0,
        cache_hint=True,
    )
    payload = _anthropic_payload(request, max_tokens=128)

    history_block = payload["messages"][0]["content"][0]
    observation_content = payload["messages"][-1]["content"]

    assert history_block["cache_control"] == {"type": "ephemeral"}
    assert isinstance(observation_content, str)
    assert "cache_control" not in observation_content


def test_anthropic_payload_omits_unsupported_temperature_for_opus_4_8() -> None:
    payload = _anthropic_payload(_request("claude-opus-4-8", temperature=0.4), max_tokens=128)

    assert "temperature" not in payload


def test_anthropic_payload_keeps_temperature_for_sampling_models() -> None:
    payload = _anthropic_payload(_request("claude-sonnet-4-6", temperature=0.4), max_tokens=128)

    assert payload["temperature"] == 0.4


def test_model_alias_resolution_uses_stable_env_var(monkeypatch) -> None:
    env_var = model_alias_env_var("claude-opus-4-8")
    assert env_var == "SOLVENT_MODEL_ALIAS_CLAUDE_OPUS_4_8"
    assert resolve_model_name("claude-opus-4-8") == "claude-opus-4-8"

    monkeypatch.setenv(env_var, "claude-provider-model-id")
    assert resolve_model_name("claude-opus-4-8") == "claude-provider-model-id"


def test_harness_records_temperature_in_model_requests(tmp_path: Path) -> None:
    client = FakeClient([_response("end_tick", {}, 5, 1)])
    trace_path = tmp_path / "temperature.jsonl"
    run_episode(
        _config(trace_path),
        LLMHarness(model="fake", client=client, max_turns=1, temperature=0.7),
    )

    assert client.requests[0].temperature == 0.7
    sidecar_row = json.loads(trace_path.with_suffix(".llm.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert sidecar_row["request"]["temperature"] == 0.7


def test_harness_threads_caching_flag_into_request_and_sidecar(tmp_path: Path) -> None:
    client = FakeClient([_response("end_tick", {}, 5, 1)])
    trace_path = tmp_path / "cache-off.jsonl"
    run_episode(
        _config(trace_path),
        LLMHarness(model="fake", client=client, max_turns=1, caching=False),
    )

    assert client.requests[0].cache_hint is False
    sidecar_row = json.loads(trace_path.with_suffix(".llm.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert sidecar_row["request"]["caching"] is False
    assert sidecar_row["request"]["cache_hint"] is False


def test_harness_rejects_out_of_range_temperature() -> None:
    try:
        LLMHarness(model="fake", client=FakeClient([]), temperature=1.5)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("temperature above one should be rejected")


def _config(trace_path: Path) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="claude-opus-4-8:+procedure",
        start_balance=Decimal("1000.00"),
        horizon_ticks=1,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0"),
        trace_path=trace_path,
        delivery_mode="tool_mediated",
    )


def _response(name: str, arguments: dict, input_tokens: int, output_tokens: int) -> ModelResponse:
    return ModelResponse(
        {"name": name, "arguments": arguments},
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _request(model: str, temperature: float, cache_hint: bool = True) -> ModelRequest:
    return ModelRequest(
        model=model,
        system_prompt="system",
        observation={"tick": 0},
        tools={"end_tick": {"description": "Advance", "input_schema": {"type": "object", "properties": {}}}},
        history=[],
        temperature=temperature,
        cache_hint=cache_hint,
    )


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
