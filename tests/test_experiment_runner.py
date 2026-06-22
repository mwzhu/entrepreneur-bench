import json
import subprocess
import sys
import threading
from pathlib import Path
from decimal import Decimal

from solvent.harness.llm import BudgetExceededError
from solvent.experiment.config import experiment_config_from_dict, smoke_experiment_config
from solvent.experiment.runner import _CellOutcome, run_experiment, run_experiment_smoke
from solvent.experiment.state import COMPLETED


def test_run_experiment_writes_ledger_and_skips_completed_cells_on_resume(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: resume-smoke",
                "models: [fake:base]",
                "seeds: [1]",
                "samples_per_seed: 1",
                "conditions: [redteam_off]",
                "horizon_minutes: 1",
                "market: { arrival_rate_per_day: 1.0 }",
                "budget_usd: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    first = run_experiment(config_path, run_dir=run_dir)
    first_ledger = json.loads(first.ledger_path.read_text(encoding="utf-8"))
    first_record = first_ledger["cells"][0]

    second = run_experiment(config_path, run_dir=run_dir)
    second_ledger = json.loads(second.ledger_path.read_text(encoding="utf-8"))
    second_record = second_ledger["cells"][0]

    assert first.completed == 1
    assert second.completed == 1
    assert first_record["status"] == "completed"
    assert second_record["status"] == "completed"
    assert second_record["started_at"] == first_record["started_at"]
    assert Path(second_record["trace_path"]).exists()
    assert Path(second_record["scorecard_path"]).exists()


def test_run_experiment_default_run_dir_matches_v0_5_dod(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: dod-path-smoke",
                "models: [fake:base]",
                "seeds: [1]",
                "horizon_minutes: 1",
                "market: { arrival_rate_per_day: 1.0 }",
                "budget_usd: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_experiment(config_path)

    assert result.run_dir == Path("runs/dod-path-smoke")
    assert (tmp_path / "runs" / "dod-path-smoke" / "ledger.json").exists()


def test_experiment_records_reproducibility_provenance(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: provenance-smoke",
                "models: [fake:base]",
                "seeds: [7]",
                "samples_per_seed: 1",
                "conditions: [redteam_on]",
                "horizon_minutes: 60",
                "market: { task_mix: {data_clean: 0.5, extract: 0.5}, arrival_rate_per_day: 24.0, decoy_rate: 0.25, manipulation_rate: 0.2 }",
                "context_policy: scratchpad",
                "ctx_window_tokens: 2048",
                "caching: true",
                "temperature: 0.3",
                "budget_usd: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_experiment(config_path, run_dir=run_dir)
    ledger = json.loads(result.ledger_path.read_text(encoding="utf-8"))
    record = ledger["cells"][0]
    provenance = record["provenance"]
    first_event = json.loads(Path(record["trace_path"]).read_text(encoding="utf-8").splitlines()[0])
    trace_provenance = first_event["payload"]["provenance"]

    assert record["status"] == "completed"
    assert provenance["model"] == "fake"
    assert provenance["config_id"] == "fake:base"
    assert provenance["seed"] == 7
    assert provenance["sample_index"] == 0
    assert provenance["condition"] == "redteam_on"
    assert provenance["context_policy"] == "scratchpad"
    assert provenance["ctx_window_tokens"] == 2048
    assert provenance["caching"] is True
    assert provenance["horizon_minutes"] == 60
    assert provenance["menu_version"] == "menu_v0_4"
    assert provenance["menu_schema_version"] == "solvent_delivery_menu_v0_4"
    assert len(provenance["menu_checksum"]) == 64
    assert provenance["market"]["task_mix"]["extract"] == 0.5
    assert provenance["market"]["manipulation_rate"] == 0.2
    assert trace_provenance["brain_model"] == "fake"
    assert trace_provenance["context_policy"] == "scratchpad"
    assert trace_provenance["ctx_window_tokens"] == 2048
    assert trace_provenance["caching"] is True
    assert provenance["menu_checksum"] == trace_provenance["menu_checksum"]
    assert trace_provenance["manipulation_rate"] == "0.2"


def test_smoke_experiment_config_shrinks_matrix_to_one_cell() -> None:
    config = experiment_config_from_dict(
        {
            "name": "big",
            "models": ["fake:base", "claude-sonnet-4-6:base"],
            "seeds": [1, 2, 3],
            "samples_per_seed": 2,
            "conditions": ["redteam_off", "redteam_on"],
            "ablations": ["base", "+procedure"],
            "horizon_minutes": 1440,
            "market": {"arrival_rate_per_day": 1.0},
            "budget_usd": 100,
            "parallelism": 8,
        }
    )

    smoke = smoke_experiment_config(config, model="claude-sonnet-4-6:base", budget_usd=2, horizon_minutes=30)

    assert smoke.name == "big_smoke"
    assert smoke.models == ["claude-sonnet-4-6:base"]
    assert smoke.seeds == [1]
    assert smoke.samples_per_seed == 1
    assert smoke.conditions == ["redteam_off"]
    assert smoke.ablations == ["base"]
    assert smoke.horizon_minutes == 30
    assert smoke.market.arrival_rate_per_day == 48.0
    assert smoke.budget_usd == 2
    assert smoke.parallelism == 1
    assert smoke.cell_count == 1


def test_smoke_experiment_config_can_keep_all_models_for_readiness_smoke() -> None:
    config = experiment_config_from_dict(
        {
            "name": "big",
            "models": ["fake:base", "stub:happy_path"],
            "seeds": [1, 2],
            "samples_per_seed": 2,
            "conditions": ["redteam_off", "redteam_on"],
            "ablations": ["base", "+procedure"],
            "horizon_minutes": 1440,
            "market": {"arrival_rate_per_day": 1.0},
            "budget_usd": 100,
            "parallelism": 8,
        }
    )

    smoke = smoke_experiment_config(config, all_models=True, budget_usd=3, horizon_minutes=45)

    assert smoke.models == ["fake:base", "stub:happy_path"]
    assert smoke.seeds == [1]
    assert smoke.samples_per_seed == 1
    assert smoke.conditions == ["redteam_off"]
    assert smoke.ablations == ["base"]
    assert smoke.horizon_minutes == 45
    assert smoke.market.arrival_rate_per_day == 32.0
    assert smoke.budget_usd == 3
    assert smoke.parallelism == 1
    assert smoke.cell_count == 2


def test_smoke_experiment_config_rejects_single_model_and_all_models_mix() -> None:
    config = experiment_config_from_dict({"name": "big", "models": ["fake:base"], "seeds": [1]})

    try:
        smoke_experiment_config(config, model="fake:base", all_models=True)
    except ValueError as exc:
        assert "either a single smoke model or all_models" in str(exc)
    else:
        raise AssertionError("smoke should reject mutually exclusive model selectors")


def test_smoke_experiment_config_rejects_model_outside_matrix() -> None:
    config = experiment_config_from_dict({"name": "big", "models": ["fake:base"], "seeds": [1]})

    try:
        smoke_experiment_config(config, model="gpt-5.4-mini:base")
    except ValueError as exc:
        assert "not in experiment config" in str(exc)
    else:
        raise AssertionError("smoke model outside matrix should fail")


def test_run_experiment_smoke_uses_same_runner_state_and_provenance(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: smoke-source",
                "models: [fake:base]",
                "seeds: [10, 11]",
                "samples_per_seed: 2",
                "conditions: [redteam_off, redteam_on]",
                "horizon_minutes: 1440",
                "market: { arrival_rate_per_day: 1.0 }",
                "budget_usd: 100",
                "parallelism: 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_experiment_smoke(config_path, run_dir=run_dir, budget_usd=1, horizon_minutes=60)
    ledger = json.loads(result.ledger_path.read_text(encoding="utf-8"))
    record = ledger["cells"][0]

    assert result.name == "smoke-source_smoke"
    assert result.total_cells == 1
    assert result.completed == 1
    assert record["provenance"]["seed"] == 10
    assert record["provenance"]["condition"] == "redteam_off"
    assert record["provenance"]["horizon_minutes"] == 60
    assert record["provenance"]["market"]["arrival_rate_per_day"] == 24.0


def test_cli_experiment_smoke_runs_one_cell(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: cli-smoke-source",
                "models: [fake:base]",
                "seeds: [1, 2]",
                "samples_per_seed: 2",
                "conditions: [redteam_off, redteam_on]",
                "horizon_minutes: 1440",
                "budget_usd: 100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "experiment",
            "smoke",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["name"] == "cli-smoke-source_smoke"
    assert payload["total_cells"] == 1
    assert payload["completed"] == 1


def test_cli_experiment_smoke_all_models_runs_one_cell_per_model(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: cli-all-smoke-source",
                "models: [fake:base, stub:happy_path]",
                "seeds: [1, 2]",
                "samples_per_seed: 2",
                "conditions: [redteam_off, redteam_on]",
                "horizon_minutes: 1440",
                "budget_usd: 100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "experiment",
            "smoke",
            str(config_path),
            "--all-models",
            "--run-dir",
            str(run_dir),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    ledger = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stderr
    assert payload["name"] == "cli-all-smoke-source_smoke"
    assert payload["total_cells"] == 2
    assert payload["completed"] == 2
    assert {cell["cell"]["model"] for cell in ledger["cells"]} == {"fake:base", "stub:happy_path"}


def test_cli_experiment_smoke_returns_nonzero_when_cell_does_not_complete(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: cli-smoke-budget",
                "models: [claude-opus-4-8:base]",
                "seeds: [1]",
                "horizon_minutes: 1440",
                "budget_usd: 100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "experiment",
            "smoke",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--budget-usd",
            "0.000001",
            "--yes",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["completed"] == 0
    assert payload["skipped_budget"] == 1


def test_cli_experiment_run_skips_budget_cells_before_live_model_call(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: budget-smoke",
                "models: [claude-opus-4-8:base]",
                "seeds: [1]",
                "samples_per_seed: 1",
                "conditions: [redteam_off]",
                "horizon_minutes: 1440",
                "market: { arrival_rate_per_day: 1.0 }",
                "budget_usd: 0.000001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "experiment",
            "run",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--yes",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    ledger = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert payload["completed"] == 0
    assert payload["skipped_budget"] == 1
    assert ledger["cells"][0]["status"] == "skipped_budget"
    assert not (run_dir / "traces").exists()


def test_run_experiment_marks_runaway_cell_failed_budget(tmp_path: Path, monkeypatch) -> None:
    class BudgetHarness:
        def run(self, env) -> None:
            env._emit(
                "brain_metered",
                {
                    "model": "claude-opus-4-8",
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": "0.010000",
                    "cumulative_input_tokens": 1000,
                    "cumulative_output_tokens": 100,
                    "cumulative_cache_read_tokens": 0,
                    "cumulative_cache_write_tokens": 0,
                    "cumulative_cost": "0.010000",
                    "ablations": [],
                },
                0,
            )
            raise BudgetExceededError("cell budget exceeded")

    monkeypatch.setattr("solvent.experiment.runner._harness_for_cell", lambda config, cell, budget_limit=None: BudgetHarness())
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: failed-budget-smoke",
                "models: [claude-opus-4-8:base]",
                "seeds: [1]",
                "samples_per_seed: 1",
                "conditions: [redteam_off]",
                "horizon_minutes: 1",
                "market: { arrival_rate_per_day: 1.0 }",
                "budget_usd: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_experiment(config_path, run_dir=run_dir)
    ledger = json.loads(result.ledger_path.read_text(encoding="utf-8"))
    record = ledger["cells"][0]

    assert result.completed == 0
    assert result.failed_budget == 1
    assert record["status"] == "failed_budget"
    assert record["actual_cost"] == "0.010000"
    assert Path(record["trace_path"]).exists()
    assert Path(record["scorecard_path"]).exists()


def test_sequential_run_allows_cell_to_use_remaining_budget_above_estimate(tmp_path: Path, monkeypatch) -> None:
    seen_budget_limits: list[Decimal | None] = []

    def fake_execute(config, cell, trace_path, budget_limit):
        seen_budget_limits.append(budget_limit)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("{}\n", encoding="utf-8")
        scorecard_path = trace_path.with_suffix(".scorecard.json")
        scorecard_path.write_text("{}\n", encoding="utf-8")
        return _CellOutcome(COMPLETED, Decimal("0.080000"), trace_path, scorecard_path)

    monkeypatch.setattr("solvent.experiment.runner._execute_cell", fake_execute)
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: sequential-headroom-smoke",
                "models: [claude-sonnet-4-6:base]",
                "seeds: [1]",
                "samples_per_seed: 1",
                "conditions: [redteam_off]",
                "horizon_minutes: 10",
                "market: { arrival_rate_per_day: 144.0 }",
                "budget_usd: 1",
                "parallelism: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_experiment(config_path, run_dir=run_dir)
    ledger = json.loads(result.ledger_path.read_text(encoding="utf-8"))
    record = ledger["cells"][0]

    assert seen_budget_limits == [Decimal("1.0")]
    assert result.completed == 1
    assert result.failed_budget == 0
    assert record["status"] == "completed"
    assert Decimal(record["actual_cost"]) > Decimal(record["estimated_cost"])


def test_run_experiment_launches_cells_in_parallel(tmp_path: Path, monkeypatch) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()
    both_started = threading.Event()

    def fake_execute(config, cell, trace_path, budget_limit):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                both_started.set()
        assert both_started.wait(2), "second cell was not launched concurrently"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("{}\n", encoding="utf-8")
        scorecard_path = trace_path.with_suffix(".scorecard.json")
        scorecard_path.write_text("{}\n", encoding="utf-8")
        with lock:
            active -= 1
        return _CellOutcome(COMPLETED, Decimal("0"), trace_path, scorecard_path)

    monkeypatch.setattr("solvent.experiment.runner._execute_cell", fake_execute)
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: parallel-smoke",
                "models: [fake:base]",
                "seeds: [1, 2]",
                "samples_per_seed: 1",
                "conditions: [redteam_off]",
                "horizon_minutes: 1",
                "market: { arrival_rate_per_day: 1.0 }",
                "budget_usd: 1",
                "parallelism: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_experiment(config_path, run_dir=run_dir)

    assert result.completed == 2
    assert max_active == 2


def test_parallel_reservations_prevent_budget_overshoot(tmp_path: Path, monkeypatch) -> None:
    launched = 0

    def fake_execute(config, cell, trace_path, budget_limit):
        nonlocal launched
        launched += 1
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("{}\n", encoding="utf-8")
        scorecard_path = trace_path.with_suffix(".scorecard.json")
        scorecard_path.write_text("{}\n", encoding="utf-8")
        return _CellOutcome(COMPLETED, Decimal("0"), trace_path, scorecard_path)

    monkeypatch.setattr("solvent.experiment.runner._execute_cell", fake_execute)
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: reservation-smoke",
                "models: [claude-opus-4-8:base]",
                "seeds: [1, 2]",
                "samples_per_seed: 1",
                "conditions: [redteam_off]",
                "horizon_minutes: 1440",
                "market: { arrival_rate_per_day: 1.0 }",
                "budget_usd: 0.15",
                "parallelism: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_experiment(config_path, run_dir=run_dir, yes=True)
    ledger = json.loads(result.ledger_path.read_text(encoding="utf-8"))

    assert launched == 1
    assert result.completed == 1
    assert result.skipped_budget == 1
    assert sorted(cell["status"] for cell in ledger["cells"]) == ["completed", "skipped_budget"]
