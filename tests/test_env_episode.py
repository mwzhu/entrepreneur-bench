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
    assert summary.end_balance > summary.start_balance
    assert summary.net_revenue > 0
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


def test_rejected_counter_leaves_floor_open_and_blocks_submit_or_rebid(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "rejected.jsonl"))
    try:
        job = env.list_jobs()[0]
        # A counter above the hidden ceiling is rejected, not auto-declined.
        result = env.bid(job.id, Decimal("999999.00"))
        assert result["accepted"] is False
        assert result["counter_rejected"] is True
        # The job leaves the main board but is surfaced for an accept/decline decision.
        assert job.id not in [public.id for public in env.list_jobs()]
        assert job.id in [public.id for public in (j for j in env.awaiting_decision_jobs())]
        # Cannot submit (unaccepted) and cannot counter again.
        with pytest.raises(InvalidActionError):
            env.submit(job.id, "name,email,signup_date,plan\n")
        with pytest.raises(InvalidActionError):
            env.bid(job.id, Decimal("0.01"))
    finally:
        summary = env.finalize()
    events = _events(summary.trace_path)
    kinds = [event["kind"] for event in events]
    assert kinds.count("counter_rejected") == 1
    assert kinds.count("bid_declined") == 0
    assert kinds.count("invalid_action") == 2


def test_accept_floor_after_rejected_counter_uses_starting_price(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "accept-floor.jsonl"))
    try:
        job = env.list_jobs()[0]
        starting = next(j for j in env.market.all_jobs() if j.id == job.id).starting_price
        env.bid(job.id, Decimal("999999.00"))
        result = env.accept(job.id)
        assert result["accepted"] is True
        assert result["contract_price"] == starting
        assert env.accepted_jobs[job.id].contract_price == starting
    finally:
        summary = env.finalize()
    kinds = [event["kind"] for event in _events(summary.trace_path)]
    assert kinds.count("counter_rejected") == 1
    assert kinds.count("job_accepted") == 1


def test_accept_open_job_takes_starting_price(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "accept-open.jsonl"))
    try:
        job = env.list_jobs()[0]
        starting = next(j for j in env.market.all_jobs() if j.id == job.id).starting_price
        result = env.accept(job.id)
        assert result["accepted"] is True
        assert result["contract_price"] == starting
        # Accepting again is invalid.
        with pytest.raises(InvalidActionError):
            env.accept(job.id)
        assert job.id not in [public.id for public in env.list_jobs()]
    finally:
        summary = env.finalize()
    events = _events(summary.trace_path)
    accepted = next(e for e in events if e["kind"] == "job_accepted")
    assert accepted["payload"]["counter_accepted"] is False
    assert accepted["payload"]["contract_price"] == str(starting)


def test_decline_removes_job_and_blocks_further_action(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "decline.jsonl"))
    try:
        job = env.list_jobs()[0]
        result = env.decline(job.id)
        assert result["declined"] is True
        assert job.id not in [public.id for public in env.list_jobs()]
        with pytest.raises(InvalidActionError):
            env.accept(job.id)
        with pytest.raises(InvalidActionError):
            env.bid(job.id, Decimal("0.50"))
    finally:
        summary = env.finalize()
    kinds = [event["kind"] for event in _events(summary.trace_path)]
    assert kinds.count("job_declined") == 1
    assert kinds.count("invalid_action") == 2


def test_winning_counter_below_reservation_pays_counter_price(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "win-counter.jsonl"))
    try:
        job = env.list_jobs()[0]
        internal = next(j for j in env.market.all_jobs() if j.id == job.id)
        counter = internal.starting_price + Decimal("0.01")
        result = env.bid(job.id, counter)
        assert result["accepted"] is True
        assert env.accepted_jobs[job.id].contract_price == counter
    finally:
        summary = env.finalize()
    events = _events(summary.trace_path)
    accepted = next(e for e in events if e["kind"] == "bid_accepted")
    assert accepted["payload"]["counter_accepted"] is True
    assert accepted["payload"]["contract_price"] == str(counter)
    assert accepted["payload"]["counter_price"] == str(counter)


def test_counter_below_starting_price_normalizes_to_floor_accept(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "below-floor.jsonl"))
    try:
        job = env.list_jobs()[0]
        internal = next(j for j in env.market.all_jobs() if j.id == job.id)
        result = env.bid(job.id, Decimal("0.50"))
        assert result["accepted"] is True
        assert result["normalized_from_below_floor"] is True
        assert env.accepted_jobs[job.id].contract_price == internal.starting_price
    finally:
        summary = env.finalize()
    accepted = next(e for e in _events(summary.trace_path) if e["kind"] == "job_accepted")
    assert accepted["payload"]["normalized_from_below_floor"] is True
    assert accepted["payload"]["contract_price"] == str(internal.starting_price)


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
    assert "internal_difficulty" not in raw_trace
    assert "pass_prob" not in raw_trace


def test_finalize_is_idempotent_and_terminated_event_emitted_once(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "idempotent.jsonl"))
    try:
        env.end_tick()
    finally:
        first = env.finalize()
        second = env.finalize()
    assert first == second
    assert [event["kind"] for event in _events(first.trace_path)].count("terminated") == 1


def test_finalize_debits_breach_fee_once_before_terminated(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "breach.jsonl", breach_fee_frac=Decimal("0.25")))
    try:
        job = env.list_jobs()[0]
        accepted = env.accept(job.id)
        contract_price = Decimal(str(accepted["contract_price"]))
    finally:
        first = env.finalize()
        second = env.finalize()

    events = _events(first.trace_path)
    breach_events = [event for event in events if event["kind"] == "breach"]
    assert first == second
    assert len(breach_events) == 1
    assert events[-2]["kind"] == "breach"
    assert events[-1]["kind"] == "terminated"
    assert breach_events[0]["payload"]["fee"] == str((contract_price * Decimal("0.25")).quantize(Decimal("0.01")))
    assert [event["kind"] for event in events].count("terminated") == 1


def test_finalize_relabels_breach_caused_insolvency(tmp_path: Path) -> None:
    env = Environment(
        _config(
            tmp_path / "breach-insolvent.jsonl",
            start_balance=Decimal("1.00"),
            tool_call_cost=Decimal("0"),
            breach_fee_frac=Decimal("10.00"),
        )
    )
    try:
        job = env.list_jobs()[0]
        env.accept(job.id)
    finally:
        summary = env.finalize()

    assert summary.terminated_reason == "insolvent"
    assert summary.end_balance < 0
    assert _events(summary.trace_path)[-1]["payload"]["reason"] == "insolvent"


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


def _config(
    trace_path: Path,
    *,
    start_balance: Decimal = Decimal("20.00"),
    tool_call_cost: Decimal = Decimal("0.01"),
    breach_fee_frac: Decimal = Decimal("0"),
) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="stub:happy_path",
        start_balance=start_balance,
        horizon_ticks=3,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=tool_call_cost,
        trace_path=trace_path,
        delivery_mode="direct",
        breach_fee_frac=breach_fee_frac,
    )


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
