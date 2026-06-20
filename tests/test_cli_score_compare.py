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
    assert summary["seeds"] == [42, 43]
    assert "stub:naive" in summary["configs"]
    assert "paired_delta" in summary
    assert "net_revenue" in summary["paired_delta"]
