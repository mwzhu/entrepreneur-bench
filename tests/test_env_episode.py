import json
from decimal import Decimal
from pathlib import Path

import pytest

from solvent.cli.main import run_episode
from solvent.env.env import Environment
from solvent.env.errors import InvalidActionError
from solvent.env.models import EnvConfig
from solvent.harness.stub import StubHarness


def test_happy_path_stub_produces_paid_job_and_expected_balance(tmp_path: Path) -> None:
    summary = run_episode(_config(tmp_path / "happy.jsonl"), StubHarness("happy_path"))
    assert summary.terminated_reason == "turn_cap"
    assert summary.jobs_paid == 1
    assert summary.end_balance == Decimal("20.29")
    assert summary.net_revenue == Decimal("0.29")
    kinds = [event["kind"] for event in _events(summary.trace_path)]
    assert "verified_pass" in kinds
    assert kinds.count("paid") == 1
    assert kinds.count("terminated") == 1


def test_bad_delivery_fails_verification_and_no_revenue(tmp_path: Path) -> None:
    summary = run_episode(_config(tmp_path / "bad.jsonl"), StubHarness("bad_delivery"))
    assert summary.jobs_paid == 0
    assert summary.end_balance == Decimal("19.79")
    kinds = [event["kind"] for event in _events(summary.trace_path)]
    assert "verified_fail" in kinds
    assert "paid" not in kinds


def test_declined_bid_cannot_be_submitted_rebid_or_seen_on_board(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "declined.jsonl"))
    try:
        job = env.list_jobs()[0]
        assert env.bid(job.id, Decimal("999.00"))["accepted"] is False
        assert job.id not in [public.id for public in env.list_jobs()]
        with pytest.raises(InvalidActionError):
            env.submit(job.id, "name,email,signup_date,plan\n")
        with pytest.raises(InvalidActionError):
            env.bid(job.id, Decimal("0.01"))
    finally:
        summary = env.finalize()
    events = _events(summary.trace_path)
    assert [event["kind"] for event in events].count("bid_declined") == 1
    assert [event["kind"] for event in events].count("invalid_action") == 2


def test_successful_job_cannot_be_paid_twice(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "paid-once.jsonl"))
    try:
        job = env.list_jobs()[0]
        inspected = env.inspect_job(job.id)
        assert env.bid(job.id, Decimal("0.50"))["accepted"]
        from solvent.tasks.data_clean import build_clean_csv

        env.submit(job.id, build_clean_csv(inspected.inputs["csv"]))
        with pytest.raises(InvalidActionError):
            env.submit(job.id, build_clean_csv(inspected.inputs["csv"]))
    finally:
        summary = env.finalize()
    events = _events(summary.trace_path)
    assert [event["kind"] for event in events].count("paid") == 1


def test_trace_is_jsonl_and_does_not_leak_hidden_fields(tmp_path: Path) -> None:
    summary = run_episode(_config(tmp_path / "trace.jsonl"), StubHarness("happy_path"))
    events = _events(summary.trace_path)
    assert events[0]["kind"] == "episode_started"
    assert events[-1]["kind"] == "terminated"
    submitted = next(event for event in events if event["kind"] == "submitted")
    assert submitted["payload"]["artifact_preview"].startswith("name,email,signup_date,plan")
    assert submitted["payload"]["artifact_size"] >= len(submitted["payload"]["artifact_preview"])
    assert len(submitted["payload"]["artifact_sha256"]) == 64
    assert submitted["payload"]["artifact_truncated"] is False
    raw_trace = summary.trace_path.read_text(encoding="utf-8")
    assert "reservation_price" not in raw_trace
    assert "est_cost" not in raw_trace
    assert "rubric" not in raw_trace
    assert "true_value" not in raw_trace
    assert "is_decoy" not in raw_trace


def test_finalize_is_idempotent_and_terminated_event_emitted_once(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "idempotent.jsonl"))
    try:
        env.end_tick()
    finally:
        first = env.finalize()
        second = env.finalize()
    assert first == second
    assert [event["kind"] for event in _events(first.trace_path)].count("terminated") == 1


def test_invalid_agent_action_emits_invalid_action_with_burn(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "invalid.jsonl"))
    try:
        with pytest.raises(InvalidActionError):
            env.submit("missing", "artifact")
    finally:
        summary = env.finalize()
    invalid = [event for event in _events(summary.trace_path) if event["kind"] == "invalid_action"]
    assert len(invalid) == 1
    assert invalid[0]["burn_delta"] == "0.01"


def _config(trace_path: Path) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="stub:happy_path",
        start_balance=Decimal("20.00"),
        horizon_ticks=3,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0.01"),
        trace_path=trace_path,
    )


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
