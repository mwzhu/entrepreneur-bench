import json
from decimal import Decimal
from pathlib import Path

from solvent.env.env import Environment
from solvent.env.models import EnvConfig


def test_work_duration_advances_business_time_when_enabled(tmp_path: Path) -> None:
    env = Environment(
        _config(
            tmp_path / "work-time.jsonl",
            work_time_enabled=True,
            difficulty_distribution={"hard": 1.0},
        )
    )
    try:
        job = env.list_jobs()[0]
        assert env.bid(job.id, Decimal("0.50"))["accepted"]
        env.deliver(job.id, "tool-pro")
    finally:
        summary = env.finalize()

    events = _events(summary.trace_path)
    delivery_ticks = [
        event for event in events if event["kind"] == "tick_advanced" and event["payload"].get("reason") == "delivery_work"
    ]
    delivery_overhead = [
        event for event in events if event["kind"] == "overhead_charged" and event["payload"].get("reason") == "delivery_work"
    ]
    assert len(delivery_ticks) == 2
    assert len(delivery_overhead) == 2
    assert summary.ticks_elapsed == 2


def test_jobs_expire_after_ttl_when_enabled(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "expiry.jsonl", job_ttl_ticks=1, horizon_ticks=2))
    try:
        assert env.list_jobs()
        env.end_tick()
        assert env.list_jobs() == []
    finally:
        env.finalize()


def test_reputation_gates_high_value_jobs_when_enabled(tmp_path: Path) -> None:
    low_rep = Environment(
        _config(
            tmp_path / "low-reputation.jsonl",
            reputation_enabled=True,
            reputation_start=Decimal("0.50"),
        )
    )
    try:
        assert low_rep.list_jobs() == []
    finally:
        low_rep.finalize()

    high_rep = Environment(
        _config(
            tmp_path / "high-reputation.jsonl",
            reputation_enabled=True,
            reputation_start=Decimal("1.00"),
        )
    )
    try:
        assert high_rep.list_jobs()
    finally:
        high_rep.finalize()


def test_reputation_changes_after_delivery_when_enabled(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "reputation-change.jsonl", reputation_enabled=True))
    try:
        job = env.list_jobs()[0]
        assert env.bid(job.id, Decimal("0.50"))["accepted"]
        env.deliver(job.id, "tool-pro")
    finally:
        summary = env.finalize()

    reputation_events = [event for event in _events(summary.trace_path) if event["kind"] == "reputation_changed"]
    assert len(reputation_events) == 1
    assert reputation_events[0]["payload"]["job_id"] == "dc-42-0"
    assert reputation_events[0]["payload"]["reputation"] != reputation_events[0]["payload"]["previous"]


def _config(trace_path: Path, **overrides) -> EnvConfig:
    values = {
        "seed": 42,
        "config_id": "stub:test",
        "start_balance": Decimal("20.00"),
        "horizon_ticks": 5,
        "overhead_per_tick": Decimal("0.05"),
        "tool_call_cost": Decimal("0.01"),
        "trace_path": trace_path,
        "market_version": "data_clean_static_v0_2",
        "market_size": 1,
        "decoy_rate": Decimal("0"),
        "delivery_mode": "tool_mediated",
    }
    values.update(overrides)
    return EnvConfig(**values)


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
