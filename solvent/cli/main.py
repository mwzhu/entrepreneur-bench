from __future__ import annotations

import argparse
import json
import functools
import http.server
import socketserver
from decimal import Decimal
from pathlib import Path

from solvent.characterize import validate_menu, write_generated_menu
from solvent.cli_seed import parse_seeds, seed_split_label
from solvent.demo import CompareOptions, default_demo_options, harness_from_config_id, run_compare_artifact
from solvent.doctor import doctor, experiment_doctor
from solvent.dotenv import load_dotenv
from solvent.env.env import Environment
from solvent.env.models import EnvConfig, EpisodeSummary
from solvent.experiment.config import load_experiment_config
from solvent.experiment.estimate import estimate_experiment
from solvent.experiment.runner import result_to_json, run_experiment, run_experiment_smoke
from solvent.findings.report import generate_findings
from solvent.harness.llm import LLMHarness
from solvent.harness.model_client import RecordedClient
from solvent.harness.stub import StubHarness
from solvent.scoring.models import MetricSummary, Scorecard
from solvent.scoring.scorecard import score_trace, scorecard_to_json
from solvent.viewer.build import build_viewer
from solvent.viewer.trace_view import build_trace_view


def run_episode(config: EnvConfig, harness) -> EpisodeSummary:
    env = Environment(config)
    try:
        harness.run(env)
    finally:
        summary = env.finalize()
    return summary


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "score":
        return _score(args)
    if args.command == "replay":
        return _replay(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "demo":
        return _demo(args)
    if args.command == "view":
        return _view(args)
    if args.command == "characterize":
        return _characterize(args)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "estimate":
        return _estimate(args)
    if args.command == "experiment":
        return _experiment(args)
    if args.command == "findings":
        return _findings(args)
    parser.print_help()
    return 2


def _run(args: argparse.Namespace) -> int:
    try:
        config_id = _config_id_from_run_args(args)
        if args.job_ttl_ticks is not None and args.job_ttl_ticks < 1:
            raise ValueError("job_ttl_ticks must be at least 1")
        trace_path = args.trace_path or Path(f"runs/seed-{args.seed}-{config_id.replace(':', '-')}.jsonl")
        config = EnvConfig(
            seed=args.seed,
            config_id=config_id,
            start_balance=Decimal(args.start_balance),
            horizon_ticks=args.horizon,
            overhead_per_tick=Decimal(args.overhead),
            tool_call_cost=_run_tool_call_cost(config_id, args.tool_call_cost),
            trace_path=trace_path,
            market_version=args.market_version,
            market_size=args.market_size,
            decoy_rate=Decimal(args.decoy_rate),
            redteam_enabled=args.redteam,
            delivery_mode=_run_delivery_mode(config_id),
            work_time_enabled=args.work_time,
            job_ttl_ticks=args.job_ttl_ticks,
            reputation_enabled=args.reputation,
        )
        harness = _harness_from_run_args(config_id, args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    summary = run_episode(config, harness)
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


def _replay(args: argparse.Namespace) -> int:
    scorecard = score_trace(args.trace_path)
    view = build_trace_view(args.trace_path)
    if args.scorecard_output:
        args.scorecard_output.write_text(scorecard_to_json(scorecard) + "\n", encoding="utf-8")
    if args.view_output:
        args.view_output.write_text(json.dumps(view, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(view, sort_keys=True))
    else:
        print("Solvent replay")
        print(f"trace: {args.trace_path}")
        print(f"events: {len(view['events'])}")
        _print_scorecard(scorecard)
    return 0


def _compare(args: argparse.Namespace) -> int:
    try:
        artifact = run_compare_artifact(_compare_options_from_args(args))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
    try:
        options = default_demo_options(args.trace_dir)
        options = CompareOptions(
            config_a=args.a or options.config_a,
            config_b=args.b or options.config_b,
            seeds=parse_seeds(args.seeds) if args.seeds else options.seeds,
            trace_dir=args.trace_dir,
            redteam_paired=True,
            start_balance=args.start_balance,
            horizon=args.horizon,
            overhead=args.overhead,
            tool_call_cost=args.tool_call_cost,
            market_size=args.market_size,
            decoy_rate=args.decoy_rate,
            market_version=args.market_version,
            seed_split=seed_split_label(args.seeds),
            samples=args.samples,
            temperature=args.temperature,
            model_max_turns=args.model_max_turns,
            model_max_tokens=args.model_max_tokens,
            work_time_enabled=args.work_time,
            job_ttl_ticks=args.job_ttl_ticks,
            reputation_enabled=args.reputation,
        )
        artifact = run_compare_artifact(options)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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


def _characterize(args: argparse.Namespace) -> int:
    if not args.validate_menu and not args.generate_menu:
        raise SystemExit("characterize requires --validate-menu or --generate-menu")
    seeds = parse_seeds(args.seeds)
    profile_configs = _parse_profile_configs(args.profile_configs)
    validation = validate_menu(
        seeds=seeds if args.validate_menu else None,
        profile_configs=profile_configs,
        profile_trace_dir=args.output.parent / "validate-profile",
    )
    if args.generate_menu:
        write_generated_menu(args.output, seeds=seeds, profile_configs=profile_configs)
    if args.json:
        print(json.dumps(validation.to_dict(), sort_keys=True))
    else:
        print("Solvent characterization")
        print(f"menu: {validation.version}")
        print(f"checksum: {validation.checksum}")
        for name, passed in validation.checks.items():
            print(f"{name}: {'pass' if passed else 'fail'}")
        if validation.floor_ceiling_flags is not None:
            flags = ", ".join(validation.floor_ceiling_flags) if validation.floor_ceiling_flags else "none"
            print(f"floor_ceiling_flags: {flags}")
        if args.generate_menu:
            print(f"generated_menu: {args.output.resolve()}")
    return 0 if validation.valid else 1


def _doctor(args: argparse.Namespace) -> int:
    report = experiment_doctor(args.config, probe_live=args.probe_live) if args.config is not None else doctor(args.agent, probe_live=args.probe_live)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print("Solvent doctor")
        if "agent" in report:
            print(f"agent: {report['agent']}")
        if "config_path" in report:
            print(f"config: {report['config_path']}")
        if "name" in report:
            print(f"name: {report['name']}")
        for check in report["checks"]:
            print(f"{check['name']}: {'pass' if check['ok'] else 'fail'} ({check['detail']})")
    return 0 if report["ok"] else 1


def _estimate(args: argparse.Namespace) -> int:
    try:
        config = load_experiment_config(args.config_path)
        estimate = estimate_experiment(config)
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    payload = estimate.to_dict()
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("Solvent experiment estimate")
        print(f"name: {payload['name']}")
        print(f"total_cells: {payload['total_cells']}")
        print(f"total_cost: {payload['total_cost']}")
        print(f"budget_usd: {payload['budget_usd']}")
        print(f"over_budget: {str(payload['over_budget']).lower()}")
        for model in payload["models"]:
            print(
                f"model: {model['model']} cells={model['cells']} "
                f"cost_per_cell={model['cost_per_cell']} total={model['total_cost']}"
            )
    return 1 if estimate.over_budget and not args.yes else 0


def _experiment(args: argparse.Namespace) -> int:
    if args.experiment_command == "run":
        try:
            result = run_experiment(args.config_path, run_dir=args.run_dir, yes=args.yes)
        except (ValueError, KeyError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
        if args.json:
            print(result_to_json(result))
        else:
            payload = result.to_dict()
            print("Solvent experiment run")
            print(f"name: {payload['name']}")
            print(f"run_dir: {payload['run_dir']}")
            print(f"total_cells: {payload['total_cells']}")
            print(f"completed: {payload['completed']}")
            print(f"failed: {payload['failed']}")
            print(f"skipped_budget: {payload['skipped_budget']}")
            print(f"failed_budget: {payload['failed_budget']}")
            print(f"actual_spend: {payload['actual_spend']}")
            print(f"ledger: {payload['ledger_path']}")
        return 0
    if args.experiment_command == "smoke":
        try:
            result = run_experiment_smoke(
                args.config_path,
                run_dir=args.run_dir,
                yes=args.yes,
                model=args.model,
                all_models=args.all_models,
                budget_usd=args.budget_usd,
                horizon_minutes=args.horizon_minutes,
            )
        except (ValueError, KeyError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
        if args.json:
            print(result_to_json(result))
        else:
            payload = result.to_dict()
            print("Solvent experiment smoke")
            print(f"name: {payload['name']}")
            print(f"run_dir: {payload['run_dir']}")
            print(f"total_cells: {payload['total_cells']}")
            print(f"completed: {payload['completed']}")
            print(f"failed: {payload['failed']}")
            print(f"skipped_budget: {payload['skipped_budget']}")
            print(f"failed_budget: {payload['failed_budget']}")
            print(f"actual_spend: {payload['actual_spend']}")
            print(f"ledger: {payload['ledger_path']}")
        return 0 if result.completed == result.total_cells else 1
    raise SystemExit(f"unknown experiment command: {args.experiment_command}")


def _findings(args: argparse.Namespace) -> int:
    try:
        result = generate_findings(args.experiment_dir)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("Solvent findings")
        print(f"summary: {result['summary_path']}")
        print(f"leaderboard: {result['leaderboard_path']}")
        print(f"findings: {result['findings_path']}")
        print(f"viewer: {result['viewer_path']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solvent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run a deterministic Solvent episode")
    run_parser.add_argument("--agent", default="stub")
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
    run_parser.add_argument("--recorded-sidecar", type=Path)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--model-max-turns", type=int, default=200)
    run_parser.add_argument("--model-max-tokens", type=int, default=1024)
    run_parser.add_argument("--work-time", action="store_true", help="enable v0.4b work-duration time advancement")
    run_parser.add_argument("--job-ttl-ticks", type=int, help="expire jobs this many ticks after arrival")
    run_parser.add_argument("--reputation", action="store_true", help="enable v0.4b reputation-gated job availability")

    score_parser = subparsers.add_parser("score", help="score an existing Solvent JSONL trace")
    score_parser.add_argument("trace_path", type=Path)
    score_parser.add_argument("--json", action="store_true")
    score_parser.add_argument("--output", type=Path)

    replay_parser = subparsers.add_parser("replay", help="replay and score a saved Solvent trace without model calls")
    replay_parser.add_argument("trace_path", type=Path)
    replay_parser.add_argument("--json", action="store_true")
    replay_parser.add_argument("--scorecard-output", type=Path)
    replay_parser.add_argument("--view-output", type=Path)

    compare_parser = subparsers.add_parser("compare", help="run and score paired agent configs")
    compare_parser.add_argument("--a", required=True)
    compare_parser.add_argument("--b", required=True)
    compare_parser.add_argument("--seeds", required=True)
    compare_parser.add_argument("--horizon", type=int, default=5)
    compare_parser.add_argument("--trace-dir", type=Path, default=Path("runs/compare-v0_4"))
    compare_parser.add_argument("--redteam-paired", action="store_true")
    compare_parser.add_argument("--start-balance", default="20.00")
    compare_parser.add_argument("--overhead", default="0.05")
    compare_parser.add_argument("--tool-call-cost", default="0.01")
    compare_parser.add_argument("--market-size", type=int, default=5)
    compare_parser.add_argument("--decoy-rate", default="0.40")
    compare_parser.add_argument("--market-version", default="data_clean_static_v0_2")
    compare_parser.add_argument("--json", action="store_true")
    compare_parser.add_argument("--viewer", action="store_true")
    compare_parser.add_argument("--samples", type=int, default=1)
    compare_parser.add_argument("--temperature", type=float, default=0.0)
    compare_parser.add_argument("--model-max-turns", type=int, default=200)
    compare_parser.add_argument("--model-max-tokens", type=int, default=1024)
    compare_parser.add_argument("--work-time", action="store_true", help="enable v0.4b work-duration time advancement")
    compare_parser.add_argument("--job-ttl-ticks", type=int, help="expire jobs this many ticks after arrival")
    compare_parser.add_argument("--reputation", action="store_true", help="enable v0.4b reputation-gated job availability")

    demo_parser = subparsers.add_parser("demo", help="run the canonical v0.4 demo comparison and viewer")
    demo_parser.add_argument("--a")
    demo_parser.add_argument("--b")
    demo_parser.add_argument("--seeds")
    demo_parser.add_argument("--horizon", type=int, default=5)
    demo_parser.add_argument("--trace-dir", type=Path, default=Path("runs/demo-v0_4"))
    demo_parser.add_argument("--start-balance", default="20.00")
    demo_parser.add_argument("--overhead", default="0.05")
    demo_parser.add_argument("--tool-call-cost", default="0.01")
    demo_parser.add_argument("--market-size", type=int, default=5)
    demo_parser.add_argument("--decoy-rate", default="0.40")
    demo_parser.add_argument("--market-version", default="data_clean_static_v0_2")
    demo_parser.add_argument("--json", action="store_true")
    demo_parser.add_argument("--samples", type=int, default=1)
    demo_parser.add_argument("--temperature", type=float, default=0.0)
    demo_parser.add_argument("--model-max-turns", type=int, default=200)
    demo_parser.add_argument("--model-max-tokens", type=int, default=1024)
    demo_parser.add_argument("--work-time", action="store_true", help="enable v0.4b work-duration time advancement")
    demo_parser.add_argument("--job-ttl-ticks", type=int, help="expire jobs this many ticks after arrival")
    demo_parser.add_argument("--reputation", action="store_true", help="enable v0.4b reputation-gated job availability")

    view_parser = subparsers.add_parser("view", help="print or serve a generated Solvent viewer")
    view_parser.add_argument("trace_dir", type=Path)
    view_parser.add_argument("--serve", action="store_true")
    view_parser.add_argument("--port", type=int, default=8765)

    characterize_parser = subparsers.add_parser("characterize", help="validate or emit the frozen delivery menu")
    characterize_parser.add_argument("--validate-menu", action="store_true")
    characterize_parser.add_argument("--generate-menu", action="store_true")
    characterize_parser.add_argument("--seeds", default="dev")
    characterize_parser.add_argument("--output", type=Path, default=Path("runs/characterize/menu_v0_4.json"))
    characterize_parser.add_argument("--profile-configs", default="stub:happy_path,stub:procedure")
    characterize_parser.add_argument("--json", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="check local v0.4 readiness and live-model credentials")
    doctor_parser.add_argument("--agent", default="claude-opus-4-8:base")
    doctor_parser.add_argument("--config", type=Path, help="check a v0.5 experiment config and all referenced models")
    doctor_parser.add_argument("--probe-live", action="store_true", help="run tiny live provider probes for gateway-backed models")
    doctor_parser.add_argument("--json", action="store_true")

    estimate_parser = subparsers.add_parser("estimate", help="estimate a v0.5 experiment config before spending")
    estimate_parser.add_argument("config_path", type=Path)
    estimate_parser.add_argument("--json", action="store_true")
    estimate_parser.add_argument("--yes", action="store_true", help="return success even when estimate exceeds budget")

    experiment_parser = subparsers.add_parser("experiment", help="run and resume v0.5 experiment configs")
    experiment_subparsers = experiment_parser.add_subparsers(dest="experiment_command", required=True)
    experiment_run_parser = experiment_subparsers.add_parser("run", help="run or resume an experiment config")
    experiment_run_parser.add_argument("config_path", type=Path)
    experiment_run_parser.add_argument("--run-dir", type=Path)
    experiment_run_parser.add_argument("--json", action="store_true")
    experiment_run_parser.add_argument("--yes", action="store_true", help="allow execution when the estimate exceeds budget")
    experiment_smoke_parser = experiment_subparsers.add_parser("smoke", help="run a one-cell smoke test from an experiment config")
    experiment_smoke_parser.add_argument("config_path", type=Path)
    experiment_smoke_parser.add_argument("--model", help="model config from the experiment YAML to smoke; defaults to the first model")
    experiment_smoke_parser.add_argument("--all-models", action="store_true", help="run one smoke cell for every model in the experiment YAML")
    experiment_smoke_parser.add_argument("--run-dir", type=Path)
    experiment_smoke_parser.add_argument("--budget-usd", type=float, default=1.0)
    experiment_smoke_parser.add_argument("--horizon-minutes", type=int, default=60)
    experiment_smoke_parser.add_argument("--json", action="store_true")
    experiment_smoke_parser.add_argument("--yes", action="store_true", help="allow execution when the smoke estimate exceeds budget")

    findings_parser = subparsers.add_parser("findings", help="generate a v0.5 leaderboard, report, and viewer")
    findings_parser.add_argument("experiment_dir", type=Path)
    findings_parser.add_argument("--json", action="store_true")
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
    if scorecard.tool_selection is not None:
        print(f"tool-selection regret: {scorecard.tool_selection.oracle_tool_regret}")
    if scorecard.compute is not None:
        print(f"brain compute cost: {scorecard.compute.brain_cost}")
    print(f"manipulation conceded value: {scorecard.support.conceded_value}")
    print(f"coherence penalty: {scorecard.coherence.coherence_penalty}")
    if scorecard.compatibility_estimated_horizon:
        print("compatibility horizon estimate: true")


def _parse_profile_configs(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _config_id_from_run_args(args: argparse.Namespace) -> str:
    if args.agent == "stub":
        return f"stub:{args.stub_mode}"
    return args.agent


def _run_delivery_mode(config_id: str) -> str:
    return "direct" if config_id.startswith("stub:") else "tool_mediated"


def _run_tool_call_cost(config_id: str, configured: str) -> Decimal:
    return Decimal(configured) if config_id.startswith("stub:") else Decimal("0")


def _harness_from_run_args(config_id: str, args: argparse.Namespace):
    if args.recorded_sidecar is None:
        return harness_from_config_id(
            config_id,
            temperature=args.temperature,
            model_max_turns=args.model_max_turns,
            model_max_tokens=args.model_max_tokens,
        )
    if config_id.startswith("stub:"):
        raise SystemExit("--recorded-sidecar is only valid for LLM config ids")
    if not args.recorded_sidecar.exists():
        raise SystemExit(f"recorded sidecar not found: {args.recorded_sidecar}")
    return LLMHarness.from_config_id(
        config_id,
        client=RecordedClient(args.recorded_sidecar),
        temperature=args.temperature,
        max_turns=args.model_max_turns,
        model_max_tokens=args.model_max_tokens,
    )


def _compare_options_from_args(args: argparse.Namespace) -> CompareOptions:
    return CompareOptions(
        config_a=args.a,
        config_b=args.b,
        seeds=parse_seeds(args.seeds),
        trace_dir=args.trace_dir,
        redteam_paired=args.redteam_paired,
        start_balance=args.start_balance,
        horizon=args.horizon,
        overhead=args.overhead,
        tool_call_cost=args.tool_call_cost,
        market_size=args.market_size,
        decoy_rate=args.decoy_rate,
        market_version=args.market_version,
        seed_split=seed_split_label(args.seeds),
        samples=args.samples,
        temperature=args.temperature,
        model_max_turns=args.model_max_turns,
        model_max_tokens=args.model_max_tokens,
        work_time_enabled=args.work_time,
        job_ttl_ticks=args.job_ttl_ticks,
        reputation_enabled=args.reputation,
    )


def _print_compare(config_ids: list[str], summaries: dict, paired_delta) -> None:
    print("Solvent comparison")
    print("metric                         " + "        ".join(config_ids) + "        delta")
    for metric in [
        "net_revenue",
        "fraction_of_omniscient_optimal",
        "delivery_pass_rate",
        "brain_compute_cost",
        "tool_selection_regret",
        "pricing_regret",
        "selection_regret",
        "support_conceded_value",
        "coherence_penalty",
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
