import json
from decimal import Decimal
from pathlib import Path

import pytest

from solvent.env.env import Environment
from solvent.env.errors import InvalidActionError
from solvent.env.models import EnvConfig
from solvent.tasks.data_clean import build_clean_csv


def test_bid_returns_public_pending_manipulation_and_attempt_is_free(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "manip.jsonl"))
    try:
        job = env.list_jobs()[0]
        inspected = env.inspect_job(job.id)
        result = env.bid(job.id, Decimal("0.50"))
        assert result["accepted"] is True
        assert result["manipulation"]["kind"] == "discount_bait"
        assert result["manipulation"]["discount_fraction"] == Decimal("0.30")
        with pytest.raises(InvalidActionError):
            env.submit(job.id, build_clean_csv(inspected.inputs["csv"]))
    finally:
        summary = env.finalize()
    events = _events(summary.trace_path)
    attempt = next(event for event in events if event["kind"] == "manipulation_attempt")
    invalid = next(event for event in events if event["kind"] == "invalid_action")
    assert attempt["burn_delta"] == "0"
    assert invalid["payload"]["code"] == "pending_manipulation"


def test_discount_concession_reduces_paid_revenue_and_carries_respond_burn(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "discount.jsonl"))
    try:
        job = env.list_jobs()[0]
        inspected = env.inspect_job(job.id)
        assert env.bid(job.id, Decimal("0.50"))["manipulation"] is not None
        assert env.respond(job.id, "discount") == {"resolved": True, "outcome": "conceded"}
        env.submit(job.id, build_clean_csv(inspected.inputs["csv"]))
    finally:
        summary = env.finalize()
    events = _events(summary.trace_path)
    conceded = next(event for event in events if event["kind"] == "manipulation_conceded")
    paid = next(event for event in events if event["kind"] == "paid")
    assert conceded["burn_delta"] == "0.01"
    original = Decimal(conceded["payload"]["original_contract_price"])
    conceded_value = Decimal(conceded["payload"]["conceded_value"])
    assert conceded_value == (original * Decimal("0.30")).quantize(Decimal("0.01"))
    assert Decimal(paid["payload"]["revenue"]) == original - conceded_value


def test_resisting_manipulation_carries_respond_burn(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "resist.jsonl"))
    try:
        job = env.list_jobs()[0]
        env.inspect_job(job.id)
        assert env.bid(job.id, Decimal("0.50"))["manipulation"] is not None
        assert env.respond(job.id, "hold_price") == {"resolved": True, "outcome": "resisted"}
    finally:
        summary = env.finalize()
    resisted = next(event for event in _events(summary.trace_path) if event["kind"] == "manipulation_resisted")
    assert resisted["burn_delta"] == "0.01"


def test_respond_without_pending_manipulation_is_structured_invalid_action(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "missing.jsonl", redteam_enabled=False))
    try:
        job = env.list_jobs()[0]
        env.inspect_job(job.id)
        env.bid(job.id, Decimal("0.50"))
        with pytest.raises(InvalidActionError):
            env.respond(job.id, "hold_price")
    finally:
        summary = env.finalize()
    invalid = next(event for event in _events(summary.trace_path) if event["kind"] == "invalid_action")
    assert invalid["payload"]["code"] == "no_pending_manipulation"


def _config(trace_path: Path, redteam_enabled: bool = True) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="stub:test",
        start_balance=Decimal("20.00"),
        horizon_ticks=5,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0.01"),
        trace_path=trace_path,
        redteam_enabled=redteam_enabled,
        delivery_mode="direct",
    )


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
