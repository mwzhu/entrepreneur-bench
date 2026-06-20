from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent import __version__
from solvent.env.env import Environment
from solvent.env.models import EnvConfig
from solvent.harness.stub import StubHarness
from solvent.scoring.compare import paired_delta_scorecards, paired_delta_values, summarize_scorecards, with_manipulation_loss
from solvent.scoring.models import MetricSummary, Scorecard
from solvent.scoring.scorecard import score_trace, scorecard_to_json

DEMO_SEEDS = [40, 41, 42, 43, 44]
DEMO_CONFIG_A = "stub:naive"
DEMO_CONFIG_B = "stub:procedure"


@dataclass(frozen=True)
class CompareOptions:
    config_a: str
    config_b: str
    seeds: list[int]
    trace_dir: Path
    redteam_paired: bool = False
    start_balance: str = "20.00"
    horizon: int = 5
    overhead: str = "0.05"
    tool_call_cost: str = "0.01"
    market_size: int = 5
    decoy_rate: str = "0.40"
    market_version: str = "data_clean_static_v0_2"


@dataclass(frozen=True)
class RunArtifact:
    config_id: str
    seed: int
    redteam_enabled: bool
    trace_path: Path
    scorecard_path: Path


@dataclass(frozen=True)
class CompareArtifact:
    trace_dir: Path
    summary_path: Path
    summary: dict[str, Any]
    runs: list[RunArtifact]
    summaries: dict[str, Any]
    paired_delta: Any


def default_demo_options(trace_dir: Path) -> CompareOptions:
    return CompareOptions(
        config_a=DEMO_CONFIG_A,
        config_b=DEMO_CONFIG_B,
        seeds=list(DEMO_SEEDS),
        trace_dir=trace_dir,
        redteam_paired=True,
    )


def run_compare_artifact(options: CompareOptions) -> CompareArtifact:
    trace_dir = options.trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    config_ids = [options.config_a, options.config_b]
    off_cards: dict[str, list[Scorecard]] = {config_id: [] for config_id in config_ids}
    losses: dict[str, dict[int, float]] = {config_id: {} for config_id in config_ids}
    runs: list[RunArtifact] = []

    for config_id in config_ids:
        for seed in options.seeds:
            off_summary = _run_config_trace(config_id, seed, options, redteam=False)
            off_card = score_trace(off_summary.trace_path)
            off_cards[config_id].append(off_card)
            off_scorecard_path = off_summary.trace_path.with_suffix(".scorecard.json")
            off_scorecard_path.write_text(scorecard_to_json(off_card) + "\n", encoding="utf-8")
            runs.append(
                RunArtifact(
                    config_id=config_id,
                    seed=seed,
                    redteam_enabled=False,
                    trace_path=off_summary.trace_path,
                    scorecard_path=off_scorecard_path,
                )
            )

            if options.redteam_paired:
                on_summary = _run_config_trace(config_id, seed, options, redteam=True)
                on_card = score_trace(on_summary.trace_path)
                on_scorecard_path = on_summary.trace_path.with_suffix(".scorecard.json")
                on_scorecard_path.write_text(scorecard_to_json(on_card) + "\n", encoding="utf-8")
                runs.append(
                    RunArtifact(
                        config_id=config_id,
                        seed=seed,
                        redteam_enabled=True,
                        trace_path=on_summary.trace_path,
                        scorecard_path=on_scorecard_path,
                    )
                )
                if off_card.fraction_of_omniscient_optimal is not None and on_card.fraction_of_omniscient_optimal is not None:
                    losses[config_id][seed] = off_card.fraction_of_omniscient_optimal - on_card.fraction_of_omniscient_optimal

    summaries = {}
    for config_id in config_ids:
        summary = summarize_scorecards(off_cards[config_id])
        if options.redteam_paired:
            summary = with_manipulation_loss(summary, list(losses[config_id].values()))
        summaries[config_id] = summary

    paired_delta = paired_delta_scorecards(off_cards[options.config_a], off_cards[options.config_b])
    if options.redteam_paired:
        paired_delta = paired_delta.__class__(
            net_revenue=paired_delta.net_revenue,
            fraction_of_omniscient_optimal=paired_delta.fraction_of_omniscient_optimal,
            delivery_pass_rate=paired_delta.delivery_pass_rate,
            pricing_regret=paired_delta.pricing_regret,
            selection_regret=paired_delta.selection_regret,
            manipulation_resistance_loss=paired_delta_values(losses[options.config_a], losses[options.config_b]),
        )

    summary_payload = {
        "schema_version": "solvent_compare_v0_3",
        "created_by": f"solvent {__version__}",
        "metric_labels": {
            "net_revenue": "Net revenue (baseline, red-team off)",
            "manipulation_resistance_loss": "Manipulation-resistance loss (red-team on minus off)",
        },
        "seeds": options.seeds,
        "configs": {config_id: summary_to_dict(summary) for config_id, summary in summaries.items()},
        "paired_delta": summary_to_dict(paired_delta),
    }
    summary_path = trace_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, sort_keys=True) + "\n", encoding="utf-8")
    return CompareArtifact(
        trace_dir=trace_dir,
        summary_path=summary_path,
        summary=summary_payload,
        runs=runs,
        summaries=summaries,
        paired_delta=paired_delta,
    )


def summary_to_dict(summary: Any) -> dict[str, Any]:
    return {
        key: metric_to_dict(value)
        for key, value in summary.__dict__.items()
        if value is not None
    }


def metric_to_dict(metric: MetricSummary) -> dict[str, Any]:
    return {"mean": metric.mean, "std": metric.std, "n": metric.n}


def harness_from_config_id(config_id: str) -> StubHarness:
    family, mode = config_id.split(":", 1)
    if family != "stub":
        raise SystemExit("v0.3 only supports stub:* configs")
    return StubHarness(mode)


def _run_config_trace(config_id: str, seed: int, options: CompareOptions, redteam: bool):
    harness = harness_from_config_id(config_id)
    safe_config = config_id.replace(":", "-")
    redteam_label = "redteam-on" if redteam else "redteam-off"
    trace_path = options.trace_dir / f"seed-{seed}-{safe_config}-{redteam_label}.jsonl"
    config = EnvConfig(
        seed=seed,
        config_id=config_id,
        start_balance=Decimal(options.start_balance),
        horizon_ticks=options.horizon,
        overhead_per_tick=Decimal(options.overhead),
        tool_call_cost=Decimal(options.tool_call_cost),
        trace_path=trace_path,
        market_version=options.market_version,
        market_size=options.market_size,
        decoy_rate=Decimal(options.decoy_rate),
        redteam_enabled=redteam,
    )
    env = Environment(config)
    try:
        harness.run(env)
    finally:
        summary = env.finalize()
    return summary
