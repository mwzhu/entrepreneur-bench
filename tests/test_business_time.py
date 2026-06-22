import json
from decimal import Decimal
from pathlib import Path

from solvent.env.env import Environment
from solvent.env.models import EnvConfig
from solvent.env.tool_api import ToolAdapter


def test_business_time_delivery_advances_once_and_prorates_overhead(tmp_path: Path) -> None:
    env = Environment(
        _business_config(
            tmp_path / "delivery.jsonl",
            horizon_minutes=300,
            market_size=1,
            arrival_rate_per_day=Decimal("4.80"),
            job_ttl_minutes=300,
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
    advances = [event for event in events if event["kind"] == "business_time_advanced"]
    overhead = [event for event in events if event["kind"] == "overhead_charged" and event["payload"].get("reason") == "delivery_work"]

    assert len(advances) == 1
    assert advances[0]["payload"]["elapsed"] == 2
    assert summary.ticks_elapsed == 2
    assert len(overhead) == 1
    assert overhead[0]["payload"]["amount"] == "0.000070"


def test_advance_to_next_event_jumps_to_next_event_and_end_tick_aliases_in_business_mode(tmp_path: Path) -> None:
    env = Environment(
        _business_config(
            tmp_path / "advance.jsonl",
            horizon_minutes=1440,
            market_size=2,
            arrival_rate_per_day=Decimal("2.00"),
            job_ttl_minutes=1440,
        )
    )
    api = ToolAdapter(env)
    try:
        assert env.clock.business_time == 0
        # advance_to_next_event jumps to the next scheduled event (arrival, expiry, or
        # horizon). Arrivals are now a seeded Poisson process, so assert against the
        # env's own next_event_time rather than a fixed minute.
        expected = env.next_event_time()
        result = api.dispatch({"name": "advance_to_next_event", "arguments": {}})
        assert result["ok"] is True
        assert result["result"]["business_time"] == expected
        # end_tick is an alias for advance_to_next_event in business mode.
        if not env.terminated():
            before = env.clock.business_time
            env.end_tick()
            assert env.clock.business_time >= before
    finally:
        env.finalize()


def test_business_time_observation_contains_calendar_fields(tmp_path: Path) -> None:
    env = Environment(_business_config(tmp_path / "observe.jsonl", horizon_minutes=60))
    try:
        observation = ToolAdapter(env).observe()
    finally:
        env.finalize()

    assert observation["business_time"] == 0
    assert observation["horizon_minutes"] == 60
    assert observation["days_remaining"] == 60 / 1440
    assert observation["available_jobs"][0]["arrival_minute"] == 0
    assert observation["available_jobs"][0]["expiry_minute"] == 60


def test_finalize_distinguishes_horizon_stop_from_turn_cap(tmp_path: Path) -> None:
    env = Environment(_business_config(tmp_path / "horizon.jsonl", horizon_minutes=60))
    try:
        env.end_tick()
    finally:
        summary = env.finalize()

    assert summary.terminated_reason == "horizon"
    terminated = [event for event in _events(summary.trace_path) if event["kind"] == "terminated"]
    assert terminated[-1]["payload"]["reason"] == "horizon"


def _business_config(trace_path: Path, **overrides) -> EnvConfig:
    values = {
        "seed": 42,
        "config_id": "stub:test",
        "start_balance": Decimal("20.00"),
        "horizon_ticks": 60,
        "horizon_minutes": 60,
        "overhead_per_tick": Decimal("0.05"),
        "overhead_per_minute": Decimal("0.000035"),
        "tool_call_cost": Decimal("0.01"),
        "trace_path": trace_path,
        "market_version": "business_stream_v0_5",
        "market_size": 1,
        "arrival_rate_per_day": Decimal("1.00"),
        "decoy_rate": Decimal("0"),
        "delivery_mode": "tool_mediated",
        "job_ttl_minutes": 60,
    }
    values.update(overrides)
    return EnvConfig(**values)


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
