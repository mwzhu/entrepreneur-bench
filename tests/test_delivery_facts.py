import json
from decimal import Decimal
from pathlib import Path

import pytest

from solvent.env.env import Environment
from solvent.env.errors import InvalidActionError
from solvent.env.models import EnvConfig
from solvent.env.tool_api import ToolAdapter
from solvent.scoring.events import facts_from_events, load_events
from solvent.scoring.scorecard import score_trace


def test_deliver_path_normalizes_into_delivery_attempt_fact(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "deliver.jsonl"))
    api = ToolAdapter(env)
    try:
        listed = api.dispatch({"name": "list_jobs", "arguments": {}})
        job = listed["result"][0]
        api.dispatch({"name": "bid", "arguments": {"job_id": job["id"], "price": "0.50"}})
        delivered = api.dispatch({"name": "deliver", "arguments": {"job_id": job["id"], "model": "tool-pro"}})
        assert delivered["ok"] is True
    finally:
        summary = env.finalize()

    facts = facts_from_events(load_events(summary.trace_path))
    assert len(facts.delivery_attempts or []) == 1
    attempt = (facts.delivery_attempts or [])[0]
    assert attempt.mode == "tool_mediated"
    assert attempt.model == "tool-pro"
    assert attempt.price_charged == Decimal("45.00")
    assert attempt.attempt_index == 0

    scorecard = score_trace(summary.trace_path)
    assert scorecard.delivery.submitted_jobs == 1
    assert scorecard.delivery.passed_jobs == 1
    assert scorecard.coherence.dropped_jobs == 0
    # gross_score = dc-42-0 reservation_price * pass(1.0) under the rescaled economy.
    assert scorecard.gross_score == Decimal("384.68")
    assert scorecard.tool_selection is not None
    assert scorecard.tool_selection.tool_price_charged == Decimal("45.00")
    assert scorecard.compute is not None
    assert scorecard.compute.brain_cost == Decimal("0")


def test_deliver_rejects_unresolved_pending_manipulation(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "pending.jsonl", redteam_enabled=True))
    try:
        job = env.list_jobs()[0]
        assert env.bid(job.id, Decimal("0.50"))["manipulation"] is not None
        with pytest.raises(InvalidActionError):
            env.deliver(job.id, "tool-pro")
    finally:
        summary = env.finalize()

    events = _events(summary.trace_path)
    invalid = [event for event in events if event["kind"] == "invalid_action"]
    assert invalid[-1]["payload"]["code"] == "pending_manipulation"
    assert "delivery_passed" not in [event["kind"] for event in events]


def test_submit_and_deliver_are_gated_by_delivery_mode(tmp_path: Path) -> None:
    direct_env = Environment(_direct_config(tmp_path / "direct.jsonl"))
    try:
        job = direct_env.list_jobs()[0]
        assert direct_env.bid(job.id, Decimal("0.50"))["accepted"]
        with pytest.raises(InvalidActionError):
            direct_env.deliver(job.id, "tool-pro")
    finally:
        direct_summary = direct_env.finalize()

    tool_env = Environment(_config(tmp_path / "tool.jsonl"))
    try:
        job = tool_env.list_jobs()[0]
        assert tool_env.bid(job.id, Decimal("0.50"))["accepted"]
        with pytest.raises(InvalidActionError):
            tool_env.submit(job.id, "artifact")
    finally:
        tool_summary = tool_env.finalize()

    assert _last_invalid_code(direct_summary.trace_path) == "wrong_delivery_mode"
    assert _last_invalid_code(tool_summary.trace_path) == "wrong_delivery_mode"


def test_delivery_outcome_is_call_order_independent_for_same_job_and_model(tmp_path: Path) -> None:
    plain = _deliver_after_optional_calls(tmp_path / "plain.jsonl", extra_calls=False)
    noisy = _deliver_after_optional_calls(tmp_path / "noisy.jsonl", extra_calls=True)

    plain_delivery = _delivery_event(plain.trace_path)
    noisy_delivery = _delivery_event(noisy.trace_path)

    assert plain_delivery["kind"] == noisy_delivery["kind"]
    assert plain_delivery["payload"]["job_id"] == noisy_delivery["payload"]["job_id"]
    assert plain_delivery["payload"]["model"] == noisy_delivery["payload"]["model"]
    assert plain_delivery["payload"]["attempt_index"] == 0
    assert noisy_delivery["payload"]["attempt_index"] == 0


def _deliver_after_optional_calls(trace_path: Path, extra_calls: bool):
    env = Environment(_config(trace_path))
    try:
        job = env.list_jobs()[0]
        if extra_calls:
            env.inspect_job(job.id)
            env.check_balance()
            env.list_models()
        assert env.bid(job.id, Decimal("0.50"))["accepted"]
        if extra_calls:
            env.check_balance()
            env.list_in_progress()
        env.deliver(job.id, "tool-pro")
    finally:
        return env.finalize()


def _config(trace_path: Path, redteam_enabled: bool = False) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="stub:test",
        start_balance=Decimal("1000.00"),
        horizon_ticks=5,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0.01"),
        trace_path=trace_path,
        market_version="data_clean_static_v0_2",
        market_size=5,
        decoy_rate=Decimal("0.40"),
        redteam_enabled=redteam_enabled,
        delivery_mode="tool_mediated",
    )


def _direct_config(trace_path: Path) -> EnvConfig:
    config = _config(trace_path)
    return EnvConfig(
        seed=config.seed,
        config_id=config.config_id,
        start_balance=config.start_balance,
        horizon_ticks=config.horizon_ticks,
        overhead_per_tick=config.overhead_per_tick,
        tool_call_cost=config.tool_call_cost,
        trace_path=config.trace_path,
        market_version=config.market_version,
        market_size=config.market_size,
        decoy_rate=config.decoy_rate,
        redteam_enabled=config.redteam_enabled,
        delivery_mode="direct",
    )


def _last_invalid_code(trace_path: Path) -> str:
    return [event for event in _events(trace_path) if event["kind"] == "invalid_action"][-1]["payload"]["code"]


def _delivery_event(trace_path: Path) -> dict:
    return next(event for event in _events(trace_path) if event["kind"] in {"delivery_passed", "delivery_failed"})


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
