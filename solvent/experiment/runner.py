from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent.env.env import Environment
from solvent.env.models import EnvConfig, EpisodeSummary
from solvent.env.pricing import PRICING_TABLE_VERSION
from solvent.experiment.config import ExperimentConfig, load_experiment_config, smoke_experiment_config
from solvent.experiment.estimate import ExperimentEstimate, estimate_experiment
from solvent.experiment.matrix import ExperimentCell, expand_matrix
from solvent.experiment.state import COMPLETED, FAILED, FAILED_BUDGET, PENDING, CellRecord, ExperimentState, TERMINAL_STATUSES
from solvent.harness.llm import BudgetExceededError, LLMHarness
from solvent.harness.stub import StubHarness
from solvent.scoring.scorecard import score_trace, scorecard_to_json


@dataclass(frozen=True)
class ExperimentRunResult:
    name: str
    run_dir: Path
    total_cells: int
    completed: int
    failed: int
    skipped_budget: int
    failed_budget: int
    actual_spend: Decimal
    budget_usd: Decimal
    manifest_path: Path
    ledger_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_dir": str(self.run_dir),
            "total_cells": self.total_cells,
            "completed": self.completed,
            "failed": self.failed,
            "skipped_budget": self.skipped_budget,
            "failed_budget": self.failed_budget,
            "actual_spend": str(self.actual_spend),
            "budget_usd": str(self.budget_usd),
            "manifest_path": str(self.manifest_path),
            "ledger_path": str(self.ledger_path),
        }


@dataclass(frozen=True)
class _CellOutcome:
    status: str
    actual_cost: Decimal
    trace_path: Path
    scorecard_path: Path | None = None
    error: str | None = None


def run_experiment(config_path: Path, run_dir: Path | None = None, yes: bool = False) -> ExperimentRunResult:
    config = load_experiment_config(config_path)
    return run_experiment_config(config, run_dir=run_dir, yes=yes)


def run_experiment_smoke(
    config_path: Path,
    run_dir: Path | None = None,
    yes: bool = False,
    model: str | None = None,
    all_models: bool = False,
    budget_usd: float = 1.0,
    horizon_minutes: int = 60,
) -> ExperimentRunResult:
    config = smoke_experiment_config(
        load_experiment_config(config_path),
        model=model,
        all_models=all_models,
        budget_usd=budget_usd,
        horizon_minutes=horizon_minutes,
    )
    return run_experiment_config(config, run_dir=run_dir, yes=yes)


def run_experiment_config(config: ExperimentConfig, run_dir: Path | None = None, yes: bool = False) -> ExperimentRunResult:
    estimate = estimate_experiment(config)
    if estimate.over_budget and not yes:
        raise ValueError("estimated experiment cost exceeds budget; re-run with --yes to use the budget guard")
    cells = expand_matrix(config)
    actual_run_dir = run_dir or Path("runs") / config.name
    state = ExperimentState.load_or_create(actual_run_dir, config, estimate, cells)
    max_workers = min(config.parallelism, max(1, len(cells)))
    next_cell = 0
    in_flight: dict[Future[_CellOutcome], str] = {}

    def launch_ready(executor: ThreadPoolExecutor) -> None:
        nonlocal next_cell
        while len(in_flight) < max_workers and next_cell < len(cells):
            cell = cells[next_cell]
            next_cell += 1
            record = state.records[cell.cell_id]
            if record.status in TERMINAL_STATUSES or record.status != PENDING:
                continue
            if _would_exceed_budget(config, state, record.estimated_cost):
                state.skip_budget(cell.cell_id)
                state.save(config, estimate)
                continue
            trace_path = _trace_path(actual_run_dir, cell)
            state.start(cell.cell_id)
            state.save(config, estimate)
            budget_limit = _cell_budget_limit(config, state, record, max_workers)
            future = executor.submit(_execute_cell, config, cell, trace_path, budget_limit)
            in_flight[future] = cell.cell_id

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        launch_ready(executor)
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                cell_id = in_flight.pop(future)
                try:
                    outcome = future.result()
                except Exception as exc:  # noqa: BLE001 - protect the experiment ledger.
                    state.fail(cell_id, str(exc), _trace_path(actual_run_dir, state.records[cell_id].cell))
                else:
                    _apply_outcome(state, cell_id, outcome)
                state.save(config, estimate)
            launch_ready(executor)

    return _result(config, estimate, state)


def _execute_cell(config: ExperimentConfig, cell: ExperimentCell, trace_path: Path, budget_limit: Decimal | None) -> _CellOutcome:
    try:
        summary = _run_cell(config, cell, trace_path, budget_limit=budget_limit)
        scorecard = score_trace(summary.trace_path)
        scorecard_path = summary.trace_path.with_suffix(".scorecard.json")
        scorecard_path.write_text(scorecard_to_json(scorecard) + "\n", encoding="utf-8")
        actual_cost = scorecard.compute.brain_cost if scorecard.compute is not None else Decimal("0")
        return _CellOutcome(COMPLETED, actual_cost, summary.trace_path, scorecard_path)
    except BudgetExceededError as exc:
        scorecard = score_trace(trace_path)
        scorecard_path = trace_path.with_suffix(".scorecard.json")
        scorecard_path.write_text(scorecard_to_json(scorecard) + "\n", encoding="utf-8")
        actual_cost = scorecard.compute.brain_cost if scorecard.compute is not None else Decimal("0")
        return _CellOutcome(FAILED_BUDGET, actual_cost, trace_path, scorecard_path, str(exc))
    except Exception as exc:  # noqa: BLE001 - keep long experiment ledgers resumable.
        return _CellOutcome(FAILED, Decimal("0"), trace_path, None, str(exc))


def _apply_outcome(state: ExperimentState, cell_id: str, outcome: _CellOutcome) -> None:
    if outcome.status == COMPLETED and outcome.scorecard_path is not None:
        state.complete(cell_id, outcome.actual_cost, outcome.trace_path, outcome.scorecard_path)
        return
    if outcome.status == FAILED_BUDGET and outcome.scorecard_path is not None:
        state.fail_budget(cell_id, outcome.actual_cost, outcome.trace_path, outcome.scorecard_path, outcome.error or "cell budget exceeded")
        return
    state.fail(cell_id, outcome.error or "cell failed", outcome.trace_path)


def _run_cell(config: ExperimentConfig, cell: ExperimentCell, trace_path: Path, budget_limit: Decimal | None = None) -> EpisodeSummary:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    env_config = EnvConfig(
        seed=cell.seed,
        config_id=cell.config_id,
        start_balance=Decimal("1000.00"),
        horizon_ticks=config.horizon_minutes,
        horizon_minutes=config.horizon_minutes,
        overhead_per_tick=Decimal("10"),
        overhead_per_minute=Decimal("0.006944"),
        tool_call_cost=Decimal("0.01") if cell.config_id.startswith("stub:") else Decimal("0"),
        trace_path=trace_path,
        market_version="business_stream_v0_5",
        market_size=max(1, int(round(config.market.arrival_rate_per_day * config.horizon_minutes / 1440))),
        arrival_rate_per_day=Decimal(str(config.market.arrival_rate_per_day)),
        decoy_rate=Decimal(str(config.market.decoy_rate)),
        manipulation_rate=Decimal(str(config.market.manipulation_rate)),
        difficulty_distribution=config.market.difficulty_distribution,
        redteam_enabled=cell.redteam_enabled,
        delivery_mode="direct" if cell.config_id.startswith("stub:") else "tool_mediated",
        task_mix=config.market.task_mix,
        seed_split="experiment",
        pricing_table_version=PRICING_TABLE_VERSION,
        brain_model=cell.model_family,
        context_policy=config.context_policy,
        ctx_window_tokens=config.ctx_window_tokens,
        caching=config.caching,
        job_ttl_minutes=min(1440, config.horizon_minutes),
    )
    env = Environment(env_config)
    try:
        _harness_for_cell(config, cell, budget_limit=budget_limit).run(env)
    finally:
        summary = env.finalize()
    return summary


def _harness_for_cell(config: ExperimentConfig, cell: ExperimentCell, budget_limit: Decimal | None = None):
    family, mode = cell.config_id.split(":", 1) if ":" in cell.config_id else (cell.config_id, "base")
    if family == "stub":
        return StubHarness(mode)
    expected_jobs = max(1, round(config.market.arrival_rate_per_day * config.horizon_minutes / 1440))
    max_turns = config.max_turns or (expected_jobs * 10 + 200)
    return LLMHarness.from_config_id(
        cell.config_id,
        temperature=config.temperature,
        max_turns=max_turns,
        context_policy=config.context_policy,
        ctx_window_tokens=config.ctx_window_tokens,
        caching=config.caching,
        budget_limit=budget_limit,
    )


def _would_exceed_budget(config: ExperimentConfig, state: ExperimentState, estimated_cost: Decimal) -> bool:
    if config.budget_usd <= 0:
        return False
    budget = Decimal(str(config.budget_usd))
    return state.actual_spend + state.reserved_spend + estimated_cost > budget


def _cell_budget_limit(config: ExperimentConfig, state: ExperimentState, record: CellRecord, max_workers: int) -> Decimal | None:
    if config.budget_usd <= 0:
        return None
    if max_workers == 1:
        return max(Decimal("0"), Decimal(str(config.budget_usd)) - state.actual_spend)
    # Per-cell mid-flight cap is a runaway breaker, not aggregate budget control
    # (that is handled by the launch gate + reserved_spend). Give it generous
    # headroom over the estimate so normal cost variance (caching routing, Poisson
    # job counts) never truncates a healthy cell; only a true N-times-over runaway trips it.
    return (record.estimated_cost * Decimal(str(config.cell_budget_headroom))).quantize(Decimal("0.000001"))


def _trace_path(run_dir: Path, cell: ExperimentCell) -> Path:
    return run_dir / "traces" / f"{cell.cell_id}.jsonl"


def _result(config: ExperimentConfig, estimate: ExperimentEstimate, state: ExperimentState) -> ExperimentRunResult:
    status_counts: dict[str, int] = {}
    for record in state.records.values():
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    return ExperimentRunResult(
        name=config.name,
        run_dir=state.run_dir,
        total_cells=estimate.total_cells,
        completed=status_counts.get(COMPLETED, 0),
        failed=status_counts.get("failed", 0),
        skipped_budget=status_counts.get("skipped_budget", 0),
        failed_budget=status_counts.get("failed_budget", 0),
        actual_spend=state.actual_spend,
        budget_usd=Decimal(str(config.budget_usd)).quantize(Decimal("0.000001")),
        manifest_path=state.manifest_path,
        ledger_path=state.ledger_path,
    )


def result_to_json(result: ExperimentRunResult) -> str:
    return json.dumps(result.to_dict(), sort_keys=True)
