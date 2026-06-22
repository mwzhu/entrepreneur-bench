from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent import __version__
from solvent.env.env import Environment
from solvent.env.models import EnvConfig
from solvent.harness.llm import LLMHarness
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
    seed_split: str = "ad_hoc"
    samples: int = 1
    temperature: float = 0.0
    model_max_turns: int = 200
    model_max_tokens: int = 1024
    work_time_enabled: bool = False
    job_ttl_ticks: int | None = None
    reputation_enabled: bool = False

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be at least 1")
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("temperature must be between 0 and 1")
        if self.model_max_turns < 1:
            raise ValueError("model_max_turns must be at least 1")
        if self.model_max_tokens < 1:
            raise ValueError("model_max_tokens must be at least 1")
        if self.job_ttl_ticks is not None and self.job_ttl_ticks < 1:
            raise ValueError("job_ttl_ticks must be at least 1")


@dataclass(frozen=True)
class RunArtifact:
    config_id: str
    seed: int
    redteam_enabled: bool
    trace_path: Path
    scorecard_path: Path
    sample_index: int = 0
    cell_id: str = ""


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
    losses: dict[str, list[float]] = {config_id: [] for config_id in config_ids}
    runs: list[RunArtifact] = []

    for config_id in config_ids:
        for seed in options.seeds:
            for sample_index in range(options.samples):
                off_summary = _run_config_trace(config_id, seed, options, redteam=False, sample_index=sample_index)
                off_card = score_trace(off_summary.trace_path)
                off_cards[config_id].append(off_card)
                off_scorecard_path = off_summary.trace_path.with_suffix(".scorecard.json")
                off_scorecard_path.write_text(scorecard_to_json(off_card) + "\n", encoding="utf-8")
                runs.append(
                    RunArtifact(
                        config_id=config_id,
                        seed=seed,
                        sample_index=sample_index,
                        redteam_enabled=False,
                        trace_path=off_summary.trace_path,
                        scorecard_path=off_scorecard_path,
                    )
                )

                if options.redteam_paired:
                    on_summary = _run_config_trace(config_id, seed, options, redteam=True, sample_index=sample_index)
                    on_card = score_trace(on_summary.trace_path)
                    on_scorecard_path = on_summary.trace_path.with_suffix(".scorecard.json")
                    on_scorecard_path.write_text(scorecard_to_json(on_card) + "\n", encoding="utf-8")
                    runs.append(
                        RunArtifact(
                            config_id=config_id,
                            seed=seed,
                            sample_index=sample_index,
                            redteam_enabled=True,
                            trace_path=on_summary.trace_path,
                            scorecard_path=on_scorecard_path,
                        )
                    )
                    if off_card.fraction_of_omniscient_optimal is not None and on_card.fraction_of_omniscient_optimal is not None:
                        losses[config_id].append(off_card.fraction_of_omniscient_optimal - on_card.fraction_of_omniscient_optimal)

    summaries = {}
    for config_id in config_ids:
        summary = summarize_scorecards(off_cards[config_id])
        if options.redteam_paired:
            summary = with_manipulation_loss(summary, losses[config_id])
        summaries[config_id] = summary

    paired_delta = paired_delta_scorecards(off_cards[options.config_a], off_cards[options.config_b])
    if options.redteam_paired:
        paired_delta = paired_delta.__class__(
            net_revenue=paired_delta.net_revenue,
            fraction_of_omniscient_optimal=paired_delta.fraction_of_omniscient_optimal,
            delivery_pass_rate=paired_delta.delivery_pass_rate,
            brain_compute_cost=paired_delta.brain_compute_cost,
            tool_selection_regret=paired_delta.tool_selection_regret,
            pricing_regret=paired_delta.pricing_regret,
            selection_regret=paired_delta.selection_regret,
            support_conceded_value=paired_delta.support_conceded_value,
            coherence_penalty=paired_delta.coherence_penalty,
            manipulation_resistance_loss=paired_delta_values(_indexed_values(losses[options.config_a]), _indexed_values(losses[options.config_b])),
        )

    summary_payload = {
        "schema_version": "solvent_compare_v0_4",
        "created_by": f"solvent {__version__}",
        "metric_labels": {
            "net_revenue": "Net revenue (baseline, red-team off)",
            "brain_compute_cost": "Brain compute cost (reported only)",
            "tool_selection_regret": "Tool-selection regret",
            "support_conceded_value": "Support conceded value",
            "coherence_penalty": "Coherence penalty",
            "manipulation_resistance_loss": "Manipulation-resistance loss (red-team on minus off)",
        },
        "seeds": options.seeds,
        "samples": options.samples,
        "temperature": options.temperature,
        "model_max_turns": options.model_max_turns,
        "model_max_tokens": options.model_max_tokens,
        "work_time_enabled": options.work_time_enabled,
        "job_ttl_ticks": options.job_ttl_ticks,
        "reputation_enabled": options.reputation_enabled,
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


def harness_from_config_id(
    config_id: str,
    temperature: float = 0.0,
    model_max_turns: int = 200,
    model_max_tokens: int = 1024,
):
    family, mode = config_id.split(":", 1) if ":" in config_id else (config_id, "")
    if family == "stub":
        return StubHarness(mode)
    return LLMHarness.from_config_id(
        config_id,
        temperature=temperature,
        max_turns=model_max_turns,
        model_max_tokens=model_max_tokens,
    )


def _run_config_trace(config_id: str, seed: int, options: CompareOptions, redteam: bool, sample_index: int = 0):
    harness = harness_from_config_id(
        config_id,
        temperature=options.temperature,
        model_max_turns=options.model_max_turns,
        model_max_tokens=options.model_max_tokens,
    )
    safe_config = config_id.replace(":", "-")
    redteam_label = "redteam-on" if redteam else "redteam-off"
    sample_label = f"-sample-{sample_index}" if sample_index else ""
    trace_path = options.trace_dir / f"seed-{seed}{sample_label}-{safe_config}-{redteam_label}.jsonl"
    config = EnvConfig(
        seed=seed,
        config_id=config_id,
        start_balance=Decimal(options.start_balance),
        horizon_ticks=options.horizon,
        overhead_per_tick=Decimal(options.overhead),
        tool_call_cost=_tool_call_cost(config_id, options.tool_call_cost),
        trace_path=trace_path,
        market_version=options.market_version,
        market_size=options.market_size,
        decoy_rate=Decimal(options.decoy_rate),
        redteam_enabled=redteam,
        delivery_mode=_delivery_mode(config_id),
        seed_split=options.seed_split,
        work_time_enabled=options.work_time_enabled,
        job_ttl_ticks=options.job_ttl_ticks,
        reputation_enabled=options.reputation_enabled,
    )
    env = Environment(config)
    try:
        harness.run(env)
    finally:
        summary = env.finalize()
    return summary


def _delivery_mode(config_id: str) -> str:
    return "direct" if config_id.startswith("stub:") else "tool_mediated"


def _tool_call_cost(config_id: str, configured: str) -> Decimal:
    if not config_id.startswith("stub:"):
        return Decimal("0")
    return Decimal(configured)


def _indexed_values(values: list[float]) -> dict[int, float]:
    return {index: value for index, value in enumerate(values)}
