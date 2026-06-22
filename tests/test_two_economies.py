import json
from decimal import Decimal
from pathlib import Path

from solvent.cli.main import run_episode
from solvent.env.models import EnvConfig
from solvent.env.pricing import TokenUsage
from solvent.harness.llm import LLMHarness
from solvent.harness.model_client import FakeClient, ModelResponse
from solvent.scoring.scorecard import score_trace


def test_business_balance_debits_delivery_price_not_brain_cost(tmp_path: Path) -> None:
    trace_path = tmp_path / "two-economies.jsonl"
    client = FakeClient(
        [
            _response("list_jobs", {}, 1000, 100),
            _response("bid", {"job_id": "dc-42-0", "price": "0.50"}, 1000, 100),
            _response("deliver", {"job_id": "dc-42-0", "model": "tool-pro"}, 1000, 100),
            _response("end_tick", {}, 1000, 100),
        ]
    )

    summary = run_episode(_config(trace_path), LLMHarness("claude-opus-4-8", client=client))
    events = _events(trace_path)

    brain_cost = sum(_decimal(event["payload"]["cost"]) for event in events if event["kind"] == "brain_metered")
    assert brain_cost > Decimal("0")
    assert all(event["burn_delta"] == "0" for event in events if event["kind"] == "brain_metered")
    assert any(event["kind"] == "tool_price_charged" and event["burn_delta"] == "45.00" for event in events)

    paid_revenue = next(_decimal(event["payload"]["revenue"]) for event in events if event["kind"] == "paid")
    expected_balance = Decimal("1000.00") - Decimal("45.00") - Decimal("0.05") + paid_revenue
    assert summary.end_balance == expected_balance

    scorecard = score_trace(trace_path)
    assert scorecard.compute is not None
    assert scorecard.compute.brain_cost == brain_cost
    assert scorecard.tool_selection is not None
    assert scorecard.tool_selection.tool_price_charged == Decimal("45.00")


def test_llm_config_uses_zero_flat_tool_call_cost(tmp_path: Path) -> None:
    trace_path = tmp_path / "flat-cost.jsonl"
    client = FakeClient([_response("end_tick", {}, 10, 5)])

    run_episode(_config(trace_path), LLMHarness("fake", client=client))
    first = _events(trace_path)[0]

    assert first["payload"]["tool_call_cost"] == "0"


def _config(trace_path: Path) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="claude-opus-4-8:base",
        start_balance=Decimal("1000.00"),
        horizon_ticks=1,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0"),
        trace_path=trace_path,
        delivery_mode="tool_mediated",
    )


def _response(name: str, arguments: dict, input_tokens: int, output_tokens: int) -> ModelResponse:
    return ModelResponse({"name": name, "arguments": arguments}, TokenUsage(input_tokens, output_tokens))


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]


def _decimal(value) -> Decimal:
    return Decimal(str(value))
