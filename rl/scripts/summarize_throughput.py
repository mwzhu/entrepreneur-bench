from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent.scoring.scorecard import score_trace


@dataclass(frozen=True)
class EpisodeMeasurement:
    trace_path: Path
    turns: int
    wall_seconds: float
    model_seconds: float
    input_tokens: int
    output_tokens: int
    expected_net_revenue: Decimal
    selection_regret: Decimal
    pricing_regret: Decimal
    oracle_tool_regret: Decimal


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize M0.5 throughput smoke results.")
    parser.add_argument("results_dir", type=Path, help="Directory containing vf-eval results.jsonl.")
    parser.add_argument("--hourly-rate", type=Decimal, required=True, help="Node hourly rate in USD.")
    parser.add_argument("--node-spec", required=True, help="Human-readable node spec, e.g. '1x A6000 spot'.")
    parser.add_argument("--budget", type=Decimal, default=Decimal("300"), help="Training budget envelope in USD.")
    parser.add_argument("--group-size", type=int, default=8, help="GRPO group size G.")
    parser.add_argument("--min-episodes", type=int, default=64, help="Minimum episodes required for the M0.5 gate.")
    parser.add_argument("--output", type=Path, default=Path("rl/artifacts/throughput_smoke.md"))
    args = parser.parse_args()

    outputs = _load_outputs(args.results_dir / "results.jsonl")
    measurements = [_measurement(row) for row in outputs if row.get("trace_path")]
    errors = [row for row in outputs if row.get("error")]
    if not measurements:
        raise SystemExit("No scored outputs with trace_path found.")

    wall_total = sum(item.wall_seconds for item in measurements)
    turns_total = sum(item.turns for item in measurements)
    episodes = len(measurements)
    dollars_total = args.hourly_rate * Decimal(str(wall_total / 3600))
    dollars_per_episode = dollars_total / Decimal(episodes)
    dollars_per_1k = dollars_per_episode * Decimal("1000")
    affordable_episodes = int(args.budget / dollars_per_episode) if dollars_per_episode > 0 else 0
    affordable_step_batch = affordable_episodes // args.group_size if args.group_size > 0 else 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        _markdown(
            node_spec=args.node_spec,
            hourly_rate=args.hourly_rate,
            results_dir=args.results_dir,
            measurements=measurements,
            errors=errors,
            wall_total=wall_total,
            turns_total=turns_total,
            dollars_per_1k=dollars_per_1k,
            affordable_episodes=affordable_episodes,
            affordable_step_batch=affordable_step_batch,
            group_size=args.group_size,
            budget=args.budget,
            min_episodes=args.min_episodes,
        ),
        encoding="utf-8",
    )
    print(args.output)


def _load_outputs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing results file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _measurement(row: dict[str, Any]) -> EpisodeMeasurement:
    timing = row.get("timing") or {}
    token_usage = row.get("token_usage") or {}
    trace_path = Path(str(row["trace_path"]))
    scorecard = score_trace(trace_path)
    return EpisodeMeasurement(
        trace_path=trace_path,
        turns=_turn_count(row),
        wall_seconds=float(timing.get("total") or (timing.get("generation") or {}).get("duration") or 0.0),
        model_seconds=float((timing.get("model") or {}).get("duration") or 0.0),
        input_tokens=int(token_usage.get("input_tokens") or scorecard.compute.brain_tokens_in),
        output_tokens=int(token_usage.get("output_tokens") or scorecard.compute.brain_tokens_out),
        expected_net_revenue=scorecard.expected_net_revenue,
        selection_regret=scorecard.selection.selection_regret,
        pricing_regret=scorecard.pricing.pricing_regret,
        oracle_tool_regret=scorecard.tool_selection.oracle_tool_regret,
    )


def _turn_count(row: dict[str, Any]) -> int:
    completion = row.get("completion") or []
    return sum(1 for message in completion if message.get("role") == "assistant")


def _markdown(
    *,
    node_spec: str,
    hourly_rate: Decimal,
    results_dir: Path,
    measurements: list[EpisodeMeasurement],
    errors: list[dict[str, Any]],
    wall_total: float,
    turns_total: int,
    dollars_per_1k: Decimal,
    affordable_episodes: int,
    affordable_step_batch: int,
    group_size: int,
    budget: Decimal,
    min_episodes: int,
) -> str:
    turns = [item.turns for item in measurements]
    wall = [item.wall_seconds for item in measurements]
    turns_per_second = turns_total / wall_total if wall_total > 0 else 0.0
    mean_wall = statistics.mean(wall)
    mean_turns = statistics.mean(turns)
    p95_wall = sorted(wall)[max(0, int(len(wall) * 0.95) - 1)]
    scored_rows = "\n".join(
        "| {trace} | {turns} | {wall:.2f} | {tokens_in} | {tokens_out} | {net} | {selection} | {pricing} | {tool} |".format(
            trace=item.trace_path,
            turns=item.turns,
            wall=item.wall_seconds,
            tokens_in=item.input_tokens,
            tokens_out=item.output_tokens,
            net=item.expected_net_revenue,
            selection=item.selection_regret,
            pricing=item.pricing_regret,
            tool=item.oracle_tool_regret,
        )
        for item in measurements[:20]
    )
    error_note = "None" if not errors else f"{len(errors)} rollout(s) had errors; inspect results.jsonl."
    status = "measured" if len(measurements) >= min_episodes and not errors else "partial measurement"
    gate_note = (
        "T1 satisfied."
        if len(measurements) >= min_episodes and not errors
        else f"T1 not satisfied: need at least {min_episodes} clean episodes."
    )
    return f"""# M0.5 Throughput Smoke

Status: {status}

## Setup

- Results directory: `{results_dir}`
- Node spec: {node_spec}
- Hourly rate: ${hourly_rate}/hr
- Episodes scored: {len(measurements)}
- Errors: {error_note}
- Gate note: {gate_note}

## Throughput

- Total model turns: {turns_total}
- Aggregate wall time: {wall_total:.2f}s
- Turns/sec: {turns_per_second:.3f}
- Mean turns/episode: {mean_turns:.2f}
- Mean wall-clock/episode: {mean_wall:.2f}s
- P95 wall-clock/episode: {p95_wall:.2f}s
- Cost per 1k C1 episodes: ${dollars_per_1k.quantize(Decimal("0.01"))}

## Budget Derivation

- Budget envelope: ${budget}
- Affordable C1 episodes at measured rate: {affordable_episodes}
- With GRPO group size G={group_size}, `max_steps x batch_size <= {affordable_step_batch}` before other budget slices.
- Suggested split: reserve ~60% for C1-C2, ~20% for C3+sweep, ~20% for baselines+eval; update this section after choosing the actual training schedule.

## Scored Sample

| Trace | Turns | Wall s | Input tokens | Output tokens | Expected net | Selection regret | Pricing regret | Tool regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{scored_rows}
"""


if __name__ == "__main__":
    main()
