import json
from decimal import Decimal
from pathlib import Path

from solvent.env.env import Environment
from solvent.env.models import EnvConfig
from solvent.scoring.scorecard import score_trace


def test_episode_started_includes_v0_4_provenance(tmp_path: Path) -> None:
    env = Environment(
        EnvConfig(
            seed=42,
            config_id="stub:test",
            start_balance=Decimal("20.00"),
            horizon_ticks=1,
            overhead_per_tick=Decimal("0.05"),
            tool_call_cost=Decimal("0.01"),
            trace_path=tmp_path / "trace.jsonl",
            delivery_mode="tool_mediated",
            seed_split="dev",
        )
    )
    summary = env.finalize()

    first = json.loads(summary.trace_path.read_text(encoding="utf-8").splitlines()[0])
    provenance = first["payload"]["provenance"]
    assert provenance["seed"] == 42
    assert provenance["delivery_mode"] == "tool_mediated"
    assert provenance["menu_version"] == "menu_v0_4"
    assert len(provenance["menu_checksum"]) == 64
    assert provenance["seed_split"] == "dev"
    assert provenance["pricing_table_version"] == "pricing_v0_4"
    assert provenance["menu_schema_version"] == "solvent_delivery_menu_v0_4"

    scorecard = score_trace(summary.trace_path)
    assert scorecard.delivery_mode == "tool_mediated"
    assert scorecard.seed_split == "dev"
    assert scorecard.menu_checksum == provenance["menu_checksum"]


def test_score_trace_fails_loudly_on_menu_checksum_mismatch(tmp_path: Path) -> None:
    env = Environment(
        EnvConfig(
            seed=42,
            config_id="stub:test",
            start_balance=Decimal("20.00"),
            horizon_ticks=1,
            overhead_per_tick=Decimal("0.05"),
            tool_call_cost=Decimal("0"),
            trace_path=tmp_path / "trace.jsonl",
            delivery_mode="tool_mediated",
        )
    )
    summary = env.finalize()
    text = summary.trace_path.read_text(encoding="utf-8").replace('"menu_checksum":"', '"menu_checksum":"bad')
    summary.trace_path.write_text(text, encoding="utf-8")

    try:
        score_trace(summary.trace_path)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("checksum mismatch should fail loudly")
