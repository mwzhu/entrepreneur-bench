import json
import subprocess
import sys
from pathlib import Path


def test_cli_demo_writes_default_viewer_and_all_runs(tmp_path: Path) -> None:
    trace_dir = tmp_path / "demo"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "demo",
            "--trace-dir",
            str(trace_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (trace_dir / "summary.json").exists()
    assert (trace_dir / "manifest.json").exists()
    assert (trace_dir / "viewer" / "index.html").exists()
    assert (trace_dir / "viewer" / "data.js").exists()
    assert len(list(trace_dir.glob("*.jsonl"))) == 20

    manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["runs"]) == 20
    assert any(run["redteam_enabled"] for run in manifest["runs"])
    assert any(not run["redteam_enabled"] for run in manifest["runs"])

    summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["metric_labels"]["net_revenue"] == "Net revenue (baseline, red-team off)"
    assert summary["paired_delta"]["net_revenue"]["mean"] >= 0.50
    assert summary["configs"]["stub:naive"]["manipulation_resistance_loss"]["n"] == 5
    naive_conceded = 0
    procedure_conceded = 0
    for path in trace_dir.glob("*redteam-on.scorecard.json"):
        scorecard = json.loads(path.read_text(encoding="utf-8"))
        if "stub-naive" in path.name:
            naive_conceded += scorecard["support"]["manipulation_conceded"]
        if "stub-procedure" in path.name:
            procedure_conceded += scorecard["support"]["manipulation_conceded"]
    assert naive_conceded >= 1
    assert procedure_conceded == 0

    view_result = subprocess.run(
        [sys.executable, "-m", "solvent.cli.main", "view", str(trace_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert view_result.returncode == 0
    assert "viewer:" in view_result.stdout


def test_cli_compare_viewer_writes_view_artifacts(tmp_path: Path) -> None:
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
            "42",
            "--trace-dir",
            str(trace_dir),
            "--redteam-paired",
            "--viewer",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (trace_dir / "viewer" / "index.html").exists()
    assert (trace_dir / "viewer" / "data.js").exists()
    assert (trace_dir / "manifest.json").exists()
