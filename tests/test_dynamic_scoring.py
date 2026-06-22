import json
from pathlib import Path

from solvent.delivery.menu import DeliveryMenu
from solvent.scoring.scorecard import score_trace


def test_winnable_expired_job_counts_as_selection_regret(tmp_path: Path) -> None:
    trace_path = _write_business_trace(tmp_path / "winnable.jsonl", job_ttl_minutes=60)

    scorecard = score_trace(trace_path)

    assert scorecard.selection.good_available == 1
    assert scorecard.selection.good_chosen == 0
    assert scorecard.selection.selection_regret > 0
    assert scorecard.fraction_of_realizable is not None


def test_exogenously_unreachable_expired_job_is_not_charged_to_selection(tmp_path: Path) -> None:
    trace_path = _write_business_trace(tmp_path / "unreachable.jsonl", job_ttl_minutes=0)

    scorecard = score_trace(trace_path)

    assert scorecard.selection.good_available == 0
    assert scorecard.selection.selection_regret == 0
    assert scorecard.fraction_of_realizable is None


def test_large_business_stream_labels_reference_relaxation(tmp_path: Path) -> None:
    trace_path = _write_large_business_trace(tmp_path / "large.jsonl")

    scorecard = score_trace(trace_path)

    assert scorecard.omniscient_reference_relaxation is True
    assert scorecard.realizable_reference_relaxation is True
    assert scorecard.selection.good_available > 16


def _write_business_trace(path: Path, job_ttl_minutes: int) -> Path:
    menu_checksum = DeliveryMenu.load_default().checksum
    events = [
        {
            "tick": 0,
            "kind": "episode_started",
            "payload": {
                "seed": 42,
                "config_id": "fake:base",
                "start_balance": "20.00",
                "horizon_ticks": 60,
                "horizon_minutes": 60,
                "business_time": 0,
                "overhead_per_tick": "0.05",
                "overhead_per_minute": "0.000035",
                "tool_call_cost": "0",
                "market_version": "business_stream_v0_5",
                "market_size": 1,
                "arrival_rate_per_day": "24.0",
                "decoy_rate": "0",
                "redteam_enabled": False,
                "provenance": {
                    "seed": 42,
                    "market_version": "business_stream_v0_5",
                    "market_size": 1,
                    "decoy_rate": "0",
                    "menu_version": "menu_v0_4",
                    "menu_checksum": menu_checksum,
                    "delivery_mode": "tool_mediated",
                    "task_mix": {"data_clean": 1.0},
                    "difficulty_distribution": {"easy": 1.0},
                    "seed_split": "test",
                    "pricing_table_version": "pricing_v0_5_2026_06_20",
                    "corpus_schema_version": "none",
                    "menu_schema_version": "solvent_delivery_menu_v0_4",
                    "work_time_enabled": True,
                    "job_ttl_minutes": job_ttl_minutes,
                    "business_time_mode": True,
                    "reputation_enabled": False,
                },
            },
            "balance_after": "20.00",
            "burn_delta": "0",
        },
        {
            "tick": 60,
            "kind": "terminated",
            "payload": {"reason": "turn_cap"},
            "balance_after": "20.00",
            "burn_delta": "0",
        },
    ]
    path.write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n", encoding="utf-8")
    return path


def _write_large_business_trace(path: Path) -> Path:
    menu_checksum = DeliveryMenu.load_default().checksum
    events = [
        {
            "tick": 0,
            "kind": "episode_started",
            "payload": {
                "seed": 42,
                "config_id": "fake:base",
                "start_balance": "20.00",
                "horizon_ticks": 43200,
                "horizon_minutes": 43200,
                "business_time": 0,
                "overhead_per_tick": "0.05",
                "overhead_per_minute": "0.000035",
                "tool_call_cost": "0",
                "market_version": "business_stream_v0_5",
                "market_size": 30,
                "arrival_rate_per_day": "1.0",
                "decoy_rate": "0",
                "redteam_enabled": False,
                "provenance": {
                    "seed": 42,
                    "market_version": "business_stream_v0_5",
                    "market_size": 30,
                    "decoy_rate": "0",
                    "menu_version": "menu_v0_4",
                    "menu_checksum": menu_checksum,
                    "delivery_mode": "tool_mediated",
                    "task_mix": {"data_clean": 1.0},
                    "difficulty_distribution": {"easy": 1.0},
                    "seed_split": "test",
                    "pricing_table_version": "pricing_v0_5_2026_06_20",
                    "corpus_schema_version": "none",
                    "menu_schema_version": "solvent_delivery_menu_v0_4",
                    "work_time_enabled": True,
                    "job_ttl_minutes": 1440,
                    "business_time_mode": True,
                    "reputation_enabled": False,
                },
            },
            "balance_after": "20.00",
            "burn_delta": "0",
        },
        {
            "tick": 43200,
            "kind": "terminated",
            "payload": {"reason": "turn_cap"},
            "balance_after": "20.00",
            "burn_delta": "0",
        },
    ]
    path.write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n", encoding="utf-8")
    return path
