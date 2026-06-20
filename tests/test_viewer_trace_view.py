import json
from decimal import Decimal
from pathlib import Path

from solvent.cli.main import run_episode
from solvent.env.models import EnvConfig
from solvent.harness.stub import StubHarness
from solvent.scoring.scorecard import score_trace, scorecard_to_json
from solvent.viewer.trace_view import build_trace_view


def test_build_trace_view_extracts_public_timeline_artifact_and_scorecard(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    summary = run_episode(_config(trace_path, redteam_enabled=True), StubHarness("naive"))
    scorecard = score_trace(summary.trace_path)
    scorecard_path = tmp_path / "trace.scorecard.json"
    scorecard_path.write_text(scorecard_to_json(scorecard) + "\n", encoding="utf-8")

    view = build_trace_view(summary.trace_path, scorecard_path, root_dir=tmp_path)

    assert view["seed"] == 42
    assert view["config_id"] == "stub:test"
    assert view["redteam_enabled"] is True
    assert view["trace_path"] == "trace.jsonl"
    assert view["scorecard"]["trace_path"] == "trace.jsonl"
    assert len(view["balance_curve"]) == len(view["events"])
    assert view["jobs"]

    submitted = next(event for event in view["events"] if event["kind"] == "submitted")
    assert submitted["artifact_preview"].startswith("name,email,signup_date,plan")
    assert submitted["artifact_size"] >= len(submitted["artifact_preview"])
    assert len(submitted["artifact_sha256"]) == 64
    assert submitted["verify"]["passed"] is True
    assert submitted["verify"]["checks"]

    raw = json.dumps(view)
    assert "reservation_price" not in raw
    assert "est_cost" not in raw
    assert "rubric" not in raw
    assert "true_value" not in raw
    assert "is_decoy" not in raw
    assert str(tmp_path) not in raw


def _config(trace_path: Path, redteam_enabled: bool = False) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="stub:test",
        start_balance=Decimal("20.00"),
        horizon_ticks=5,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0.01"),
        trace_path=trace_path,
        redteam_enabled=redteam_enabled,
    )
