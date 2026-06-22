import json
from decimal import Decimal
from pathlib import Path

from solvent.cli.main import run_episode
from solvent.env.models import EnvConfig
from solvent.harness.stub import StubHarness
from solvent.scoring.scorecard import score_trace


def test_happy_path_stub_trace_semantics_stay_compatible(tmp_path: Path) -> None:
    summary = run_episode(_config(tmp_path / "happy.jsonl", horizon=3), StubHarness("happy_path"))
    events = _events(summary.trace_path)
    kinds = [event["kind"] for event in events]

    assert kinds == [
        "episode_started",
        "board_seen",
        "inspected",
        "bid_made",
        "bid_accepted",
        "submitted",
        "verified_pass",
        "paid",
        "overhead_charged",
        "tick_advanced",
        "board_seen",
        "overhead_charged",
        "tick_advanced",
        "board_seen",
        "overhead_charged",
        "tick_advanced",
        "terminated",
    ]
    assert summary.end_balance > summary.start_balance
    scorecard = score_trace(summary.trace_path)
    assert scorecard.delivery.pass_rate == 1.0
    assert scorecard.coherence.coherence_penalty == Decimal("0.00")


def test_invalid_loop_stub_still_surfaces_duplicate_bid_loop(tmp_path: Path) -> None:
    summary = run_episode(_config(tmp_path / "invalid.jsonl"), StubHarness("invalid_loop"))
    events = _events(summary.trace_path)
    invalids = [event for event in events if event["kind"] == "invalid_action"]

    assert [event["payload"]["code"] for event in invalids] == ["duplicate_bid", "duplicate_bid", "duplicate_bid"]
    scorecard = score_trace(summary.trace_path)
    assert scorecard.coherence.duplicate_bid_attempts == 3
    assert scorecard.coherence.action_loops == 1


def _config(trace_path: Path, horizon: int = 5) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="stub:compat",
        start_balance=Decimal("20.00"),
        horizon_ticks=horizon,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0.01"),
        trace_path=trace_path,
        delivery_mode="direct",
    )


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
