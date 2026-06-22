import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from solvent.cli.main import run_episode
from solvent.env.models import EnvConfig
from solvent.env.pricing import TokenUsage
from solvent.experiment.config import experiment_config_from_dict, load_experiment_config
from solvent.experiment.estimate import calibrate_estimate_against_recorded_cost, estimate_experiment
from solvent.harness.llm import LLMHarness
from solvent.harness.model_client import FakeClient, ModelResponse
from solvent.scoring.scorecard import score_trace


def test_load_experiment_config_parses_doc_style_yaml(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "\n".join(
            [
                "name: smoke",
                "models: [fake:base, claude-opus-4-8:base]",
                "seeds: [1, 2]",
                "samples_per_seed: 2",
                "conditions: [redteam_off, redteam_on]",
                "horizon_minutes: 1440",
                "market: { task_mix: {data_clean: 0.5, extract: 0.5}, arrival_rate_per_day: 2.0, decoy_rate: 0.25 }",
                "context_policy: sliding_window",
                "ctx_window_tokens: 1024",
                "ablations: [base, +procedure]",
                "caching: true",
                "budget_usd: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_experiment_config(path)

    assert config.name == "smoke"
    assert config.models == ["fake:base", "claude-opus-4-8:base"]
    assert config.seeds == [1, 2]
    assert config.cell_count == 32
    assert config.market.task_mix["extract"] == 0.5
    assert config.market.arrival_rate_per_day == 2.0


def test_estimate_experiment_uses_known_pricing_and_cell_count(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "\n".join(
            [
                "name: estimate",
                "models: [fake:base, claude-opus-4-8:base]",
                "seeds: [1]",
                "samples_per_seed: 1",
                "conditions: [redteam_off]",
                "horizon_minutes: 1440",
                "market: { arrival_rate_per_day: 1.0 }",
                "budget_usd: 100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    estimate = estimate_experiment(load_experiment_config(path))

    assert estimate.total_cells == 2
    assert len(estimate.models) == 2
    assert estimate.models[0].model == "fake:base"
    assert estimate.models[0].total_cost == 0
    assert estimate.models[1].total_cost > 0
    assert estimate.over_budget is False


def test_estimate_unknown_model_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("name: bad\nmodels: [unknown:base]\nseeds: [1]\n", encoding="utf-8")

    try:
        estimate_experiment(load_experiment_config(path))
    except KeyError as exc:
        assert "unknown brain pricing" in str(exc)
    else:
        raise AssertionError("unknown model pricing should fail during estimate")


def test_estimator_calibrates_against_recorded_metered_run(tmp_path: Path) -> None:
    config = experiment_config_from_dict(
        {
            "name": "calibration",
            "models": ["claude-opus-4-8:base"],
            "seeds": [1],
            "horizon_minutes": 60,
            "market": {"arrival_rate_per_day": 24.0},
            "ctx_window_tokens": 2048,
            "budget_usd": 1,
        }
    )
    responses = [
        ModelResponse(
            {"name": "list_jobs", "arguments": {}},
            usage=TokenUsage(input_tokens=2048, output_tokens=160),
        )
        for _ in range(5)
    ]
    responses.append(
        ModelResponse(
            {"name": "end_tick", "arguments": {}},
            usage=TokenUsage(input_tokens=2048, output_tokens=160),
        )
    )
    summary = run_episode(
        EnvConfig(
            seed=1,
            config_id="claude-opus-4-8:base",
            start_balance=Decimal("20.00"),
            horizon_ticks=60,
            horizon_minutes=60,
            overhead_per_tick=Decimal("0.05"),
            overhead_per_minute=Decimal("0.000035"),
            tool_call_cost=Decimal("0"),
            trace_path=tmp_path / "recorded.jsonl",
            market_version="business_stream_v0_5",
            market_size=1,
            arrival_rate_per_day=Decimal("24.0"),
            delivery_mode="tool_mediated",
            brain_model="claude-opus-4-8",
            ctx_window_tokens=2048,
        ),
        LLMHarness(
            model="claude-opus-4-8",
            client=FakeClient(responses),
            max_turns=6,
            ctx_window_tokens=2048,
            caching=False,
        ),
    )
    scorecard = score_trace(summary.trace_path)

    calibration = calibrate_estimate_against_recorded_cost(
        config,
        "claude-opus-4-8:base",
        scorecard.compute.brain_cost,
        tolerance_fraction=Decimal("0.01"),
    )

    assert calibration.within_tolerance is True
    assert calibration.ratio == Decimal("1.000000")


def test_cli_estimate_json_reports_budget_status_and_fails_over_budget_without_yes(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "\n".join(
            [
                "name: cli-estimate",
                "models: [claude-opus-4-8:base]",
                "seeds: [1]",
                "horizon_minutes: 1440",
                "budget_usd: 0.000001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "solvent.cli.main", "estimate", str(path), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["name"] == "cli-estimate"
    assert payload["over_budget"] is True

    allowed = subprocess.run(
        [sys.executable, "-m", "solvent.cli.main", "estimate", str(path), "--json", "--yes"],
        check=False,
        capture_output=True,
        text=True,
    )
    allowed_payload = json.loads(allowed.stdout)

    assert allowed.returncode == 0
    assert allowed_payload["over_budget"] is True


def test_cli_estimate_text_mode_returns_success_when_under_budget(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "\n".join(
            [
                "name: cli-estimate-under",
                "models: [fake:base]",
                "seeds: [1]",
                "horizon_minutes: 60",
                "budget_usd: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "solvent.cli.main", "estimate", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "over_budget: false" in result.stdout
    assert "model: fake:base" in result.stdout
