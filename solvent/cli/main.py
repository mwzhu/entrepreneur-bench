from __future__ import annotations

import argparse
import json
import functools
import http.server
import socketserver
from decimal import Decimal
from pathlib import Path

from solvent.demo import CompareOptions, default_demo_options, harness_from_config_id, run_compare_artifact
from solvent.env.env import Environment
from solvent.env.models import EnvConfig, EpisodeSummary
from solvent.harness.stub import StubHarness
from solvent.scoring.models import MetricSummary, Scorecard
from solvent.scoring.scorecard import score_trace, scorecard_to_json
from solvent.viewer.build import build_viewer


def run_episode(config: EnvConfig, harness: StubHarness) -> EpisodeSummary:
    env = Environment(config)
    try:
        harness.run(env)
    finally:
        summary = env.finalize()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "score":
        return _score(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "demo":
        return _demo(args)
    if args.command == "view":
        return _view(args)
    parser.print_help()
    return 2


def _run(args: argparse.Namespace) -> int:
    if args.agent != "stub":
        raise SystemExit("v0.2 only supports --agent stub")
    config_id = f"stub:{args.stub_mode}"
    trace_path = args.trace_path or Path(f"runs/seed-{args.seed}-stub.jsonl")
    config = EnvConfig(
        seed=args.seed,
        config_id=config_id,
        start_balance=Decimal(args.start_balance),
        horizon_ticks=args.horizon,
        overhead_per_tick=Decimal(args.overhead),
        tool_call_cost=Decimal(args.tool_call_cost),
        trace_path=trace_path,
        market_version=args.market_version,
        market_size=args.market_size,
        decoy_rate=Decimal(args.decoy_rate),
        redteam_enabled=args.redteam,
    )
    summary = run_episode(config, StubHarness(args.stub_mode))
    _print_run_summary(summary)
    if args.scorecard or args.scorecard_path:
        scorecard = score_trace(summary.trace_path)
        _print_scorecard(scorecard)
        scorecard_path = args.scorecard_path or summary.trace_path.with_suffix(".scorecard.json")
        scorecard_path.write_text(scorecard_to_json(scorecard) + "\n", encoding="utf-8")
    return 0


def _score(args: argparse.Namespace) -> int:
    scorecard = score_trace(args.trace_path)
    if args.json:
        print(scorecard_to_json(scorecard))
    else:
        _print_scorecard(scorecard)
    if args.output:
        args.output.write_text(scorecard_to_json(scorecard) + "\n", encoding="utf-8")
    return 0


def _compare(args: argparse.Namespace) -> int:
    artifact = run_compare_artifact(_compare_options_from_args(args))
    viewer_path = None
    if args.viewer:
        viewer_path = build_viewer(artifact.trace_dir, artifact.summary, artifact.runs)
    if args.json:
        print(json.dumps(artifact.summary, sort_keys=True))
    else:
        _print_compare([args.a, args.b], artifact.summaries, artifact.paired_delta)
        if viewer_path is not None:
            print(f"viewer: {viewer_path.resolve()}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    options = default_demo_options(args.trace_dir)
    options = CompareOptions(
        config_a=args.a or options.config_a,
        config_b=args.b or options.config_b,
        seeds=_parse_seeds(args.seeds) if args.seeds else options.seeds,
        trace_dir=args.trace_dir,
        redteam_paired=True,
        start_balance=args.start_balance,
        horizon=args.horizon,
        overhead=args.overhead,
        tool_call_cost=args.tool_call_cost,
        market_size=args.market_size,
        decoy_rate=args.decoy_rate,
        market_version=args.market_version,
    )
    artifact = run_compare_artifact(options)
    viewer_path = build_viewer(artifact.trace_dir, artifact.summary, artifact.runs)
    if args.json:
        print(json.dumps(artifact.summary, sort_keys=True))
    else:
        _print_compare([options.config_a, options.config_b], artifact.summaries, artifact.paired_delta)
        print(f"summary: {artifact.summary_path.resolve()}")
        print(f"viewer: {viewer_path.resolve()}")
    return 0


def _view(args: argparse.Namespace) -> int:
    viewer_path = args.trace_dir / "viewer" / "index.html"
    if not viewer_path.exists():
        raise SystemExit(f"viewer not found: {viewer_path}")
    print(f"viewer: {viewer_path.resolve()}")
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.trace_dir.resolve()))
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            print(f"serving: http://127.0.0.1:{args.port}/viewer/index.html")
            httpd.serve_forever()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solvent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run a deterministic Solvent episode")
    run_parser.add_argument("--agent", default="stub", choices=["stub"])
    run_parser.add_argument("--stub-mode", default="happy_path", choices=sorted(StubHarness.VALID_MODES))
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--start-balance", default="20.00")
    run_parser.add_argument("--horizon", type=int, default=5)
    run_parser.add_argument("--overhead", default="0.05")
    run_parser.add_argument("--tool-call-cost", default="0.01")
    run_parser.add_argument("--trace-path", type=Path)
    run_parser.add_argument("--scorecard", action="store_true")
    run_parser.add_argument("--scorecard-path", type=Path)
    run_parser.add_argument("--redteam", action="store_true")
    run_parser.add_argument("--market-size", type=int, default=5)
    run_parser.add_argument("--decoy-rate", default="0.40")
    run_parser.add_argument("--market-version", default="data_clean_static_v0_2")

    score_parser = subparsers.add_parser("score", help="score an existing Solvent JSONL trace")
    score_parser.add_argument("trace_path", type=Path)
    score_parser.add_argument("--json", action="store_true")
    score_parser.add_argument("--output", type=Path)

    compare_parser = subparsers.add_parser("compare", help="run and score paired stub configs")
    compare_parser.add_argument("--a", required=True)
    compare_parser.add_argument("--b", required=True)
    compare_parser.add_argument("--seeds", required=True)
    compare_parser.add_argument("--horizon", type=int, default=5)
    compare_parser.add_argument("--trace-dir", type=Path, default=Path("runs/compare-v0_2"))
    compare_parser.add_argument("--redteam-paired", action="store_true")
    compare_parser.add_argument("--start-balance", default="20.00")
    compare_parser.add_argument("--overhead", default="0.05")
    compare_parser.add_argument("--tool-call-cost", default="0.01")
    compare_parser.add_argument("--market-size", type=int, default=5)
    compare_parser.add_argument("--decoy-rate", default="0.40")
    compare_parser.add_argument("--market-version", default="data_clean_static_v0_2")
    compare_parser.add_argument("--json", action="store_true")
    compare_parser.add_argument("--viewer", action="store_true")

    demo_parser = subparsers.add_parser("demo", help="run the canonical v0.3 demo comparison and viewer")
    demo_parser.add_argument("--a")
    demo_parser.add_argument("--b")
    demo_parser.add_argument("--seeds")
    demo_parser.add_argument("--horizon", type=int, default=5)
    demo_parser.add_argument("--trace-dir", type=Path, default=Path("runs/demo-v0_3"))
    demo_parser.add_argument("--start-balance", default="20.00")
    demo_parser.add_argument("--overhead", default="0.05")
    demo_parser.add_argument("--tool-call-cost", default="0.01")
    demo_parser.add_argument("--market-size", type=int, default=5)
    demo_parser.add_argument("--decoy-rate", default="0.40")
    demo_parser.add_argument("--market-version", default="data_clean_static_v0_2")
    demo_parser.add_argument("--json", action="store_true")

    view_parser = subparsers.add_parser("view", help="print or serve a generated Solvent viewer")
    view_parser.add_argument("trace_dir", type=Path)
    view_parser.add_argument("--serve", action="store_true")
    view_parser.add_argument("--port", type=int, default=8765)
    return parser


def _print_run_summary(summary: EpisodeSummary) -> None:
    print("Solvent episode complete")
    print(f"seed: {summary.seed}")
    print(f"agent: {summary.config_id}")
    print(f"terminated: {summary.terminated_reason}")
    print(f"start_balance: {summary.start_balance}")
    print(f"end_balance: {summary.end_balance}")
    print(f"net: {summary.net_revenue}")
    print(f"trace: {summary.trace_path}")


def _print_scorecard(scorecard: Scorecard) -> None:
    print("Solvent scorecard")
    print(f"seed: {scorecard.seed}")
    print(f"config: {scorecard.config_id}")
    print(f"net: {scorecard.net_revenue}")
    print(f"gross: {scorecard.gross_score}")
    print(f"omniscient optimal net: {scorecard.omniscient_optimal_net}")
    print(f"fraction of omniscient optimal: {_fmt_optional(scorecard.fraction_of_omniscient_optimal)}")
    print(f"fraction of realizable: {_fmt_optional(scorecard.fraction_of_realizable)}")
    print(f"selection precision: {_fmt_optional(scorecard.selection.precision)}")
    print(f"pricing regret: {scorecard.pricing.pricing_regret}")
    print(f"delivery pass rate: {_fmt_optional(scorecard.delivery.pass_rate)}")
    print(f"manipulation conceded value: {scorecard.support.conceded_value}")
    print(f"coherence penalty: {scorecard.coherence.coherence_penalty}")
    if scorecard.compatibility_estimated_horizon:
        print("compatibility horizon estimate: true")


def _parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _compare_options_from_args(args: argparse.Namespace) -> CompareOptions:
    return CompareOptions(
        config_a=args.a,
        config_b=args.b,
        seeds=_parse_seeds(args.seeds),
        trace_dir=args.trace_dir,
        redteam_paired=args.redteam_paired,
        start_balance=args.start_balance,
        horizon=args.horizon,
        overhead=args.overhead,
        tool_call_cost=args.tool_call_cost,
        market_size=args.market_size,
        decoy_rate=args.decoy_rate,
        market_version=args.market_version,
    )


def _print_compare(config_ids: list[str], summaries: dict, paired_delta) -> None:
    print("Solvent comparison")
    print("metric                         " + "        ".join(config_ids) + "        delta")
    for metric in [
        "net_revenue",
        "fraction_of_omniscient_optimal",
        "delivery_pass_rate",
        "pricing_regret",
        "selection_regret",
        "manipulation_resistance_loss",
    ]:
        values = []
        for config_id in config_ids:
            value = getattr(summaries[config_id], metric, None)
            values.append(_fmt_summary(value) if value is not None else "n/a")
        delta = getattr(paired_delta, metric, None)
        values.append(_fmt_summary(delta) if delta is not None else "n/a")
        print(f"{metric:<30} " + "        ".join(values))


def _fmt_summary(summary: MetricSummary) -> str:
    if summary.mean is None:
        return "n/a"
    return f"{summary.mean:.2f} +/- {summary.std:.2f}"


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
