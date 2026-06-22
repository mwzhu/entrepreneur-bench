from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from solvent.env.pricing import TokenUsage, brain_cost
from solvent.experiment.config import ExperimentConfig


@dataclass(frozen=True)
class ModelEstimate:
    model: str
    cells: int
    turns_per_cell: int
    input_tokens_per_cell: int
    output_tokens_per_cell: int
    cache_read_tokens_per_cell: int
    cost_per_cell: Decimal
    total_cost: Decimal


@dataclass(frozen=True)
class ExperimentEstimate:
    name: str
    total_cells: int
    total_cost: Decimal
    budget_usd: Decimal
    over_budget: bool
    models: list[ModelEstimate]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_cells": self.total_cells,
            "total_cost": str(self.total_cost),
            "budget_usd": str(self.budget_usd),
            "over_budget": self.over_budget,
            "models": [
                {
                    "model": model.model,
                    "cells": model.cells,
                    "turns_per_cell": model.turns_per_cell,
                    "input_tokens_per_cell": model.input_tokens_per_cell,
                    "output_tokens_per_cell": model.output_tokens_per_cell,
                    "cache_read_tokens_per_cell": model.cache_read_tokens_per_cell,
                    "cost_per_cell": str(model.cost_per_cell),
                    "total_cost": str(model.total_cost),
                }
                for model in self.models
            ],
        }


@dataclass(frozen=True)
class EstimateCalibration:
    model: str
    estimated_cost: Decimal
    recorded_cost: Decimal
    ratio: Decimal | None
    within_tolerance: bool


def estimate_experiment(config: ExperimentConfig) -> ExperimentEstimate:
    jobs_over_horizon = max(1, int(round(config.market.arrival_rate_per_day * config.horizon_minutes / 1440)))
    idle_advances = max(1, jobs_over_horizon // 2)
    turns_per_cell = jobs_over_horizon * 5 + idle_advances
    context_tokens = min(config.ctx_window_tokens, 1200 + turns_per_cell * 250)
    output_tokens = turns_per_cell * 160
    uncached_input_tokens = turns_per_cell * context_tokens
    cache_read_tokens = int(uncached_input_tokens * Decimal("0.70")) if config.caching else 0
    input_tokens = uncached_input_tokens - cache_read_tokens

    cells_per_model = len(config.seeds) * config.samples_per_seed * len(config.conditions) * len(config.ablations)
    estimates = []
    total = Decimal("0")
    for model in config.models:
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
        )
        cost_per_cell = brain_cost(model.split(":", 1)[0], usage)
        model_total = (cost_per_cell * Decimal(cells_per_model)).quantize(Decimal("0.000001"))
        total += model_total
        estimates.append(
            ModelEstimate(
                model=model,
                cells=cells_per_model,
                turns_per_cell=turns_per_cell,
                input_tokens_per_cell=input_tokens,
                output_tokens_per_cell=output_tokens,
                cache_read_tokens_per_cell=cache_read_tokens,
                cost_per_cell=cost_per_cell,
                total_cost=model_total,
            )
        )
    total = total.quantize(Decimal("0.000001"))
    budget = Decimal(str(config.budget_usd)).quantize(Decimal("0.000001"))
    return ExperimentEstimate(
        name=config.name,
        total_cells=config.cell_count,
        total_cost=total,
        budget_usd=budget,
        over_budget=budget > 0 and total > budget,
        models=estimates,
    )


def calibrate_estimate_against_recorded_cost(
    config: ExperimentConfig,
    model: str,
    recorded_cost: Decimal,
    *,
    tolerance_fraction: Decimal = Decimal("0.50"),
) -> EstimateCalibration:
    """Compare the per-cell estimate to one recorded cell's metered cost."""
    estimate = estimate_experiment(config)
    try:
        model_estimate = next(item for item in estimate.models if item.model == model)
    except StopIteration as exc:
        raise ValueError(f"model is not in experiment config: {model}") from exc
    ratio = None if recorded_cost <= 0 else (model_estimate.cost_per_cell / recorded_cost).quantize(Decimal("0.000001"))
    if recorded_cost <= 0:
        within = model_estimate.cost_per_cell == 0
    else:
        lower = recorded_cost * (Decimal("1") - tolerance_fraction)
        upper = recorded_cost * (Decimal("1") + tolerance_fraction)
        within = lower <= model_estimate.cost_per_cell <= upper
    return EstimateCalibration(
        model=model,
        estimated_cost=model_estimate.cost_per_cell,
        recorded_cost=recorded_cost,
        ratio=ratio,
        within_tolerance=within,
    )
