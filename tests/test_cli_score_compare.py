import json
import subprocess
import sys
from pathlib import Path


def test_cli_run_scorecard_and_score_json(tmp_path: Path) -> None:
    trace_path = tmp_path / "cli.jsonl"
    scorecard_path = tmp_path / "cli.scorecard.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "run",
            "--agent",
            "stub",
            "--seed",
            "42",
            "--trace-path",
            str(trace_path),
            "--scorecard",
            "--scorecard-path",
            str(scorecard_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Solvent scorecard" in result.stdout
    assert scorecard_path.exists()

    score_result = subprocess.run(
        [sys.executable, "-m", "solvent.cli.main", "score", str(trace_path), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert score_result.returncode == 0
    assert json.loads(score_result.stdout)["seed"] == 42


def test_cli_compare_writes_summary(tmp_path: Path) -> None:
    trace_dir = tmp_path / "compare"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "compare",
            "--a",
            "stub:naive",
            "--b",
            "stub:procedure",
            "--seeds",
            "42,43",
            "--trace-dir",
            str(trace_dir),
            "--redteam-paired",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "solvent_compare_v0_4"
    assert summary["seeds"] == [42, 43]
    assert "stub:naive" in summary["configs"]
    assert "paired_delta" in summary
    assert "net_revenue" in summary["paired_delta"]
    assert "brain_compute_cost" in summary["paired_delta"]
    assert "tool_selection_regret" in summary["paired_delta"]
    assert "support_conceded_value" in summary["paired_delta"]
    assert "coherence_penalty" in summary["paired_delta"]


def test_cli_compare_accepts_named_seed_split(tmp_path: Path) -> None:
    trace_dir = tmp_path / "compare-dev"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "compare",
            "--a",
            "stub:happy_path",
            "--b",
            "stub:procedure",
            "--seeds",
            "dev",
            "--trace-dir",
            str(trace_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "solvent_compare_v0_4"
    assert summary["seeds"] == [40, 41, 42, 43, 44]
    first_trace = trace_dir / "seed-40-stub-happy_path-redteam-off.jsonl"
    first_event = json.loads(first_trace.read_text(encoding="utf-8").splitlines()[0])
    assert first_event["payload"]["provenance"]["seed_split"] == "dev"


def test_cli_compare_samples_repeat_each_seed_without_collapsing_delta(tmp_path: Path) -> None:
    trace_dir = tmp_path / "compare-samples"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "compare",
            "--a",
            "stub:happy_path",
            "--b",
            "stub:procedure",
            "--seeds",
            "42",
            "--samples",
            "2",
            "--temperature",
            "0.4",
            "--model-max-turns",
            "7",
            "--model-max-tokens",
            "256",
            "--work-time",
            "--job-ttl-ticks",
            "2",
            "--reputation",
            "--trace-dir",
            str(trace_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["samples"] == 2
    assert summary["temperature"] == 0.4
    assert summary["model_max_turns"] == 7
    assert summary["model_max_tokens"] == 256
    assert summary["work_time_enabled"] is True
    assert summary["job_ttl_ticks"] == 2
    assert summary["reputation_enabled"] is True
    assert summary["paired_delta"]["net_revenue"]["n"] == 2
    assert (trace_dir / "seed-42-sample-1-stub-happy_path-redteam-off.jsonl").exists()


def test_cli_compare_invalid_model_config_is_clean_error(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "compare",
            "--a",
            "fake:base+procedure",
            "--b",
            "fake:base",
            "--seeds",
            "42",
            "--trace-dir",
            str(tmp_path / "compare"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid ablation spec" in result.stderr
    assert "Traceback" not in result.stderr
