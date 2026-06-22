from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent.demo import RunArtifact
from solvent.scoring.aggregate import summarize_distribution
from solvent.scoring.scorecard import score_trace, scorecard_to_json

COMPLETED = "completed"
FAILED_BUDGET = "failed_budget"


@dataclass(frozen=True)
class FindingsData:
    experiment_dir: Path
    leaderboard: list[dict[str, Any]]
    summary: dict[str, Any]
    runs: list[RunArtifact]


def build_findings_data(experiment_dir: Path) -> FindingsData:
    experiment_dir = experiment_dir.resolve()
    ledger = _load_json(experiment_dir / "ledger.json")
    cells = ledger.get("cells", [])
    completed = [cell for cell in cells if cell.get("status") == COMPLETED]
    runs = [_run_artifact(cell) for cell in completed if cell.get("trace_path") and cell.get("scorecard_path")]
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in completed:
        groups.setdefault(_group_key(cell), []).append(cell)

    status_counts = _status_counts(cells)
    leaderboard = [
        _leaderboard_row(key, groups[key], cells)
        for key in sorted(groups)
    ]
    leaderboard.sort(key=lambda row: (row["net_revenue"]["mean"] is None, -(row["net_revenue"]["mean"] or 0), row["config_id"]))
    for index, row in enumerate(leaderboard, start=1):
        row["rank"] = index

    summary = {
        "schema_version": "solvent_findings_v0_5",
        "name": ledger.get("name", experiment_dir.name),
        "experiment_dir": str(experiment_dir),
        "status_counts": status_counts,
        "failed_cells": _failed_cells(cells),
        "metric_labels": {
            "net_revenue": "Net revenue",
            "fraction_of_omniscient_optimal": "Fraction of optimal",
            "delivery_pass_rate": "Delivery pass rate",
            "brain_compute_cost": "Brain compute cost",
            "brain_cache_read_tokens": "Brain cache-read tokens",
            "brain_cache_write_tokens": "Brain cache-write tokens",
            "brain_cache_hit_rate": "Brain cache hit rate",
            "selection_regret": "Selection regret",
            "pricing_regret": "Pricing regret",
            "tool_selection_regret": "Tool-selection regret",
            "support_conceded_value": "Support conceded value",
            "coherence_penalty": "Coherence penalty",
            "manipulation_resistance_loss": "Manipulation-resistance loss",
        },
        "configs": {
            row["config_id"]: {
                "net_revenue": row["net_revenue"],
                "fraction_of_omniscient_optimal": row["fraction_of_omniscient_optimal"],
                "omniscient_reference_relaxation": row["omniscient_reference_relaxation"],
                "realizable_reference_relaxation": row["realizable_reference_relaxation"],
                "delivery_pass_rate": row["delivery_pass_rate"],
                "brain_compute_cost": row["brain_compute_cost"],
                "brain_cache_read_tokens": row["brain_cache_read_tokens"],
                "brain_cache_write_tokens": row["brain_cache_write_tokens"],
                "brain_cache_hit_rate": row["brain_cache_hit_rate"],
                "cache_verification": row["cache_verification"],
                "selection_regret": row["selection_regret"],
                "pricing_regret": row["pricing_regret"],
                "tool_selection_regret": row["tool_selection_regret"],
                "support_conceded_value": row["support_conceded_value"],
                "coherence_penalty": row["coherence_penalty"],
                "manipulation_resistance_loss": row["manipulation_resistance_loss"],
            }
            for row in leaderboard
        },
        "leaderboard": leaderboard,
        "money_shots": _money_shots(completed),
    }
    return FindingsData(experiment_dir=experiment_dir, leaderboard=leaderboard, summary=summary, runs=runs)


def _leaderboard_row(config_id: str, cells: list[dict[str, Any]], all_cells: list[dict[str, Any]]) -> dict[str, Any]:
    scorecards = [_scorecard_for_cell(cell) for cell in cells]
    trace_events = [_events_for_cell(cell) for cell in cells]
    matching_all = [cell for cell in all_cells if _group_key(cell) == config_id]
    failed_budget = sum(1 for cell in matching_all if cell.get("status") == FAILED_BUDGET)
    total = len(matching_all)
    compute_costs = [_decimal_at(card, "compute", "brain_cost") for card in scorecards if card.get("compute") is not None]
    cache_read_values = [_nested(card, "compute", "brain_cache_read_tokens") for card in scorecards if card.get("compute") is not None]
    cache_write_values = [_nested(card, "compute", "brain_cache_write_tokens") for card in scorecards if card.get("compute") is not None]
    fraction_values = [card.get("fraction_of_omniscient_optimal") for card in scorecards]
    cache_verification = _cache_verification(matching_all, scorecards)
    manipulation_loss = _manipulation_resistance_loss(cells, scorecards)
    return {
        "config_id": config_id,
        "model": str(cells[0].get("cell", {}).get("model", config_id)),
        "completed_cells": len(cells),
        "censored_cells": failed_budget,
        "budget_kill_rate": failed_budget / total if total else 0.0,
        "net_revenue": summarize_distribution([_decimal(card.get("net_revenue")) for card in scorecards]).to_dict(),
        "fraction_of_omniscient_optimal": summarize_distribution(fraction_values).to_dict(),
        "omniscient_reference_relaxation": any(bool(card.get("omniscient_reference_relaxation")) for card in scorecards),
        "realizable_reference_relaxation": any(bool(card.get("realizable_reference_relaxation")) for card in scorecards),
        "delivery_pass_rate": summarize_distribution([_nested(card, "delivery", "pass_rate") for card in scorecards]).to_dict(),
        "jobs_delivered": summarize_distribution([_nested(card, "delivery", "passed_jobs") for card in scorecards]).to_dict(),
        "days_until_insolvent": summarize_distribution([_days_until_insolvent(events) for events in trace_events]).to_dict(),
        "horizon_fraction_active": summarize_distribution([_horizon_fraction(events) for events in trace_events]).to_dict(),
        "selection_regret": summarize_distribution([_decimal_at(card, "selection", "selection_regret") for card in scorecards]).to_dict(),
        "pricing_regret": summarize_distribution([_decimal_at(card, "pricing", "pricing_regret") for card in scorecards]).to_dict(),
        "tool_selection_regret": summarize_distribution(
            [_decimal_at(card, "tool_selection", "oracle_tool_regret") for card in scorecards if card.get("tool_selection") is not None]
        ).to_dict(),
        "support_conceded_value": summarize_distribution([_decimal_at(card, "support", "conceded_value") for card in scorecards]).to_dict(),
        "coherence_penalty": summarize_distribution([_decimal_at(card, "coherence", "coherence_penalty") for card in scorecards]).to_dict(),
        "manipulation_resistance_loss": summarize_distribution(manipulation_loss).to_dict(),
        "brain_compute_cost": summarize_distribution(compute_costs).to_dict(),
        "brain_cache_read_tokens": summarize_distribution(cache_read_values).to_dict(),
        "brain_cache_write_tokens": summarize_distribution(cache_write_values).to_dict(),
        "brain_cache_hit_rate": summarize_distribution([_cache_hit_rate(card) for card in scorecards]).to_dict(),
        "cache_verification": cache_verification,
        "efficiency": summarize_distribution(
            [
                (float(fraction) / float(cost))
                for fraction, cost in zip(fraction_values, compute_costs)
                if fraction is not None and cost is not None and cost > 0
            ]
        ).to_dict(),
    }


def _scorecard_for_cell(cell: dict[str, Any]) -> dict[str, Any]:
    scorecard_path = Path(str(cell.get("scorecard_path", "")))
    trace_path = Path(str(cell.get("trace_path", "")))
    if scorecard_path.exists():
        return _load_json(scorecard_path)
    scorecard = score_trace(trace_path)
    scorecard_path.write_text(scorecard_to_json(scorecard) + "\n", encoding="utf-8")
    return _load_json(scorecard_path)


def _events_for_cell(cell: dict[str, Any]) -> list[dict[str, Any]]:
    trace_path = Path(str(cell.get("trace_path", "")))
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_artifact(cell: dict[str, Any]) -> RunArtifact:
    cell_payload = cell.get("cell", {})
    return RunArtifact(
        config_id=str(cell_payload.get("config_id")),
        seed=int(cell_payload.get("seed")),
        sample_index=int(cell_payload.get("sample_index", 0)),
        redteam_enabled=bool(cell_payload.get("redteam_enabled", False)),
        trace_path=Path(str(cell.get("trace_path"))),
        scorecard_path=Path(str(cell.get("scorecard_path"))),
        cell_id=str(cell_payload.get("cell_id", "")),
    )


def _group_key(cell: dict[str, Any]) -> str:
    payload = cell.get("cell", {})
    return str(payload.get("config_id") or payload.get("model"))


def _status_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        status = str(cell.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _failed_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for cell in cells:
        status = str(cell.get("status", "unknown"))
        if status == COMPLETED:
            continue
        payload = cell.get("cell", {})
        failures.append(
            {
                "cell_id": str(payload.get("cell_id", "")),
                "config_id": str(payload.get("config_id") or payload.get("model", "")),
                "model": str(payload.get("model", "")),
                "status": status,
                "error": _short_error(cell.get("error")),
            }
        )
    return failures


def _short_error(error: Any) -> str:
    first_line = str(error or "").splitlines()[0] if error else ""
    return first_line[:240]


def _money_shots(cells: list[dict[str, Any]]) -> dict[str, str | None]:
    scored = [(cell, _scorecard_for_cell(cell)) for cell in cells if cell.get("scorecard_path")]
    if not scored:
        return {"max_manipulation_concession": None, "worst_coherence": None, "best_efficiency": None}
    return {
        "max_manipulation_concession": _cell_id(max(scored, key=lambda pair: _decimal_at(pair[1], "support", "conceded_value"))[0]),
        "worst_coherence": _cell_id(max(scored, key=lambda pair: _decimal_at(pair[1], "coherence", "coherence_penalty"))[0]),
        "best_efficiency": _cell_id(max(scored, key=lambda pair: _efficiency(pair[1]))[0]),
    }


def _manipulation_resistance_loss(cells: list[dict[str, Any]], scorecards: list[dict[str, Any]]) -> list[float]:
    off: dict[tuple[int, int], float] = {}
    on: dict[tuple[int, int], float] = {}
    for cell, scorecard in zip(cells, scorecards):
        fraction = scorecard.get("fraction_of_omniscient_optimal")
        if fraction is None:
            continue
        payload = cell.get("cell", {})
        key = (int(payload.get("seed", 0)), int(payload.get("sample_index", 0)))
        condition = str(payload.get("condition", ""))
        if condition == "redteam_off":
            off[key] = float(fraction)
        elif condition == "redteam_on":
            on[key] = float(fraction)
    return [off[key] - on[key] for key in sorted(set(off) & set(on))]


def _efficiency(scorecard: dict[str, Any]) -> float:
    fraction = scorecard.get("fraction_of_omniscient_optimal")
    cost = _decimal_at(scorecard, "compute", "brain_cost") if scorecard.get("compute") else Decimal("0")
    if fraction is None or cost <= 0:
        return 0.0
    return float(fraction) / float(cost)


def _cache_hit_rate(scorecard: dict[str, Any]) -> float | None:
    compute = scorecard.get("compute")
    if not isinstance(compute, dict):
        return None
    fresh = int(compute.get("brain_tokens_in") or 0)
    cached = int(compute.get("brain_cache_read_tokens") or 0)
    total = fresh + cached
    if total <= 0:
        return None
    return cached / total


def _cache_verification(cells: list[dict[str, Any]], scorecards: list[dict[str, Any]]) -> dict[str, Any]:
    provenance = [cell.get("provenance", {}) for cell in cells if isinstance(cell.get("provenance", {}), dict)]
    models = sorted(
        {
            str(cell.get("provenance", {}).get("model") or cell.get("cell", {}).get("model", ""))
            for cell in cells
        }
    )
    local = all(_is_local_model(model) for model in models) if models else False
    caching_requested = any(bool(item.get("caching")) for item in provenance)
    cache_read_tokens = sum(int(_nested(card, "compute", "brain_cache_read_tokens") or 0) for card in scorecards)
    cache_write_tokens = sum(int(_nested(card, "compute", "brain_cache_write_tokens") or 0) for card in scorecards)
    if local:
        status = "not_applicable"
        detail = "local/fake clients do not exercise provider prompt caching"
    elif not caching_requested:
        status = "not_requested"
        detail = "experiment config did not request prompt caching"
    elif cache_read_tokens > 0:
        status = "verified"
        detail = "provider reported non-zero cache-read tokens"
    else:
        status = "requested_unverified"
        detail = "caching was requested, but completed cells reported zero cache-read tokens"
    return {
        "status": status,
        "detail": detail,
        "caching_requested": caching_requested,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "models": models,
    }


def _is_local_model(model: str) -> bool:
    family = model.split(":", 1)[0]
    return family in {"fake", "recorded", "stub"} or family.startswith("stub")


def _cell_id(cell: dict[str, Any]) -> str:
    return str(cell.get("cell", {}).get("cell_id"))


def _days_until_insolvent(events: list[dict[str, Any]]) -> float | None:
    start = events[0].get("payload", {})
    horizon = start.get("horizon_minutes") or start.get("horizon_ticks")
    if horizon is None:
        return None
    last_tick = int(events[-1].get("tick", 0))
    return last_tick / 1440 if _terminated_reason(events) == "insolvent" else float(horizon) / 1440


def _horizon_fraction(events: list[dict[str, Any]]) -> float | None:
    start = events[0].get("payload", {})
    horizon = start.get("horizon_minutes") or start.get("horizon_ticks")
    if not horizon:
        return None
    return min(1.0, int(events[-1].get("tick", 0)) / float(horizon))


def _terminated_reason(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("kind") == "terminated":
            return str(event.get("payload", {}).get("reason", ""))
    return ""


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _decimal_at(data: dict[str, Any], *keys: str) -> Decimal | None:
    value = _nested(data, *keys)
    return _decimal(value)


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
