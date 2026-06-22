import json
import subprocess
import sys
from pathlib import Path

from solvent.findings.leaderboard import build_findings_data
from solvent.findings.report import generate_findings


def test_build_findings_data_excludes_failed_budget_from_capability_means(tmp_path: Path) -> None:
    run_dir = _run_tiny_experiment(tmp_path)
    ledger_path = run_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    failed = dict(ledger["cells"][0])
    failed["status"] = "failed_budget"
    failed["cell"] = dict(failed["cell"])
    failed["cell"]["cell_id"] = failed["cell"]["cell_id"] + "-failed-budget"
    ledger["cells"].append(failed)
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")

    data = build_findings_data(run_dir)
    row = data.leaderboard[0]

    assert row["completed_cells"] == 1
    assert row["censored_cells"] == 1
    assert row["budget_kill_rate"] == 0.5
    assert row["net_revenue"]["n"] == 1
    assert data.summary["status_counts"]["failed_budget"] == 1


def test_findings_surfaces_failed_cells_without_polluting_leaderboard_means(tmp_path: Path) -> None:
    run_dir = _run_tiny_experiment(tmp_path)
    ledger_path = run_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    failed = dict(ledger["cells"][0])
    failed["status"] = "failed"
    failed["actual_cost"] = "0"
    failed["scorecard_path"] = ""
    failed["trace_path"] = ""
    failed["error"] = "OPENROUTER_API_KEY rejected | details hidden"
    failed["cell"] = dict(failed["cell"])
    failed["cell"]["cell_id"] = failed["cell"]["cell_id"] + "-failed"
    failed["cell"]["model"] = "kimi-k2.6:base"
    failed["cell"]["config_id"] = "kimi-k2.6:base"
    ledger["cells"].append(failed)
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")

    result = generate_findings(run_dir)
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    findings = Path(result["findings_path"]).read_text(encoding="utf-8")

    assert summary["status_counts"]["failed"] == 1
    assert summary["failed_cells"] == [
        {
            "cell_id": failed["cell"]["cell_id"],
            "config_id": "kimi-k2.6:base",
            "model": "kimi-k2.6:base",
            "status": "failed",
            "error": "OPENROUTER_API_KEY rejected | details hidden",
        }
    ]
    assert len(summary["leaderboard"]) == 1
    assert summary["leaderboard"][0]["net_revenue"]["n"] == 1
    viewer_dir = Path(result["viewer_path"]).parent
    index_html = (viewer_dir / "index.html").read_text(encoding="utf-8")
    app_js = (viewer_dir / "app.js").read_text(encoding="utf-8")
    assert 'id="failed-cells-table"' in index_html
    assert "data.summary.failed_cells" in app_js
    assert "failed_budget" in app_js
    assert "Failed cells: 1" in findings
    assert "### Failed Cells" in findings
    assert "OPENROUTER_API_KEY rejected \\| details hidden" in findings


def test_findings_budget_censors_match_model_fallback_group(tmp_path: Path) -> None:
    run_dir = _run_tiny_experiment(tmp_path)
    ledger_path = run_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    completed = dict(ledger["cells"][0])
    completed["cell"] = dict(completed["cell"])
    completed["cell"].pop("config_id", None)
    failed = dict(completed)
    failed["status"] = "failed_budget"
    failed["cell"] = dict(failed["cell"])
    failed["cell"]["cell_id"] = failed["cell"]["cell_id"] + "-model-fallback"
    ledger["cells"] = [completed, failed]
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")

    data = build_findings_data(run_dir)
    row = data.leaderboard[0]

    assert row["config_id"] == "fake:base"
    assert row["completed_cells"] == 1
    assert row["censored_cells"] == 1
    assert row["budget_kill_rate"] == 0.5


def test_findings_computes_paired_manipulation_resistance_loss(tmp_path: Path) -> None:
    run_dir = _run_tiny_experiment(tmp_path, conditions="[redteam_off, redteam_on]")
    ledger_path = run_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for cell in ledger["cells"]:
        scorecard_path = Path(cell["scorecard_path"])
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        if cell["cell"]["condition"] == "redteam_off":
            scorecard["fraction_of_omniscient_optimal"] = 0.8
        else:
            scorecard["fraction_of_omniscient_optimal"] = 0.5
        scorecard_path.write_text(json.dumps(scorecard, sort_keys=True) + "\n", encoding="utf-8")

    result = generate_findings(run_dir)
    row = result["leaderboard"][0]
    findings = Path(result["findings_path"]).read_text(encoding="utf-8")
    app_js = (Path(result["viewer_path"]).parent / "app.js").read_text(encoding="utf-8")

    assert row["manipulation_resistance_loss"]["n"] == 1
    assert round(row["manipulation_resistance_loss"]["mean"], 6) == 0.3
    assert "manipulation loss" in findings
    assert "paired manipulation-resistance loss" in findings
    assert "row.manipulation_resistance_loss" in app_js


def test_findings_surfaces_cache_usage_metrics(tmp_path: Path) -> None:
    run_dir = _run_tiny_experiment(tmp_path)
    ledger_path = run_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["cells"][0]["cell"]["model"] = "gpt-5.4-mini:base"
    ledger["cells"][0]["cell"]["config_id"] = "gpt-5.4-mini:base"
    ledger["cells"][0]["provenance"]["model"] = "gpt-5.4-mini"
    ledger["cells"][0]["provenance"]["model_config"] = "gpt-5.4-mini:base"
    ledger["cells"][0]["provenance"]["config_id"] = "gpt-5.4-mini:base"
    ledger["cells"][0]["provenance"]["caching"] = True
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
    scorecard_path = Path(ledger["cells"][0]["scorecard_path"])
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    scorecard["config_id"] = "gpt-5.4-mini:base"
    scorecard["compute"] = {
        "brain_tokens_in": 75,
        "brain_tokens_out": 10,
        "brain_cost": "0.010000",
        "fraction_of_optimal_per_compute_dollar": 1.0,
        "brain_cache_read_tokens": 25,
        "brain_cache_write_tokens": 5,
    }
    scorecard_path.write_text(json.dumps(scorecard, sort_keys=True) + "\n", encoding="utf-8")

    result = generate_findings(run_dir)
    row = result["leaderboard"][0]

    assert row["brain_cache_read_tokens"]["mean"] == 25.0
    assert row["brain_cache_write_tokens"]["mean"] == 5.0
    assert row["brain_cache_hit_rate"]["mean"] == 0.25
    assert row["cache_verification"]["status"] == "verified"
    assert row["cache_verification"]["cache_read_tokens"] == 25
    assert "cache hit" in Path(result["findings_path"]).read_text(encoding="utf-8")
    assert "## Cache Verification" in Path(result["findings_path"]).read_text(encoding="utf-8")


def test_findings_labels_large_stream_reference_relaxation(tmp_path: Path) -> None:
    run_dir = _run_tiny_experiment(tmp_path)
    ledger = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))
    scorecard_path = Path(ledger["cells"][0]["scorecard_path"])
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    scorecard["omniscient_reference_relaxation"] = True
    scorecard["realizable_reference_relaxation"] = True
    scorecard_path.write_text(json.dumps(scorecard, sort_keys=True) + "\n", encoding="utf-8")

    result = generate_findings(run_dir)
    row = result["leaderboard"][0]
    findings = Path(result["findings_path"]).read_text(encoding="utf-8")
    viewer_js = (Path(result["viewer_path"]).parent / "app.js").read_text(encoding="utf-8")

    assert row["omniscient_reference_relaxation"] is True
    assert row["realizable_reference_relaxation"] is True
    assert "upper-bound reference" in findings
    assert "Fraction upper bound" in viewer_js
    assert "upper bound" in viewer_js


def test_findings_marks_requested_cache_without_reads_unverified(tmp_path: Path) -> None:
    run_dir = _run_tiny_experiment(tmp_path)
    ledger_path = run_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["cells"][0]["cell"]["model"] = "gpt-5.4-mini:base"
    ledger["cells"][0]["cell"]["config_id"] = "gpt-5.4-mini:base"
    ledger["cells"][0]["provenance"]["model"] = "gpt-5.4-mini"
    ledger["cells"][0]["provenance"]["model_config"] = "gpt-5.4-mini:base"
    ledger["cells"][0]["provenance"]["config_id"] = "gpt-5.4-mini:base"
    ledger["cells"][0]["provenance"]["caching"] = True
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")

    data = build_findings_data(run_dir)
    row = data.leaderboard[0]

    assert row["cache_verification"]["status"] == "requested_unverified"
    assert row["cache_verification"]["caching_requested"] is True


def test_generate_findings_writes_report_leaderboard_and_viewer(tmp_path: Path) -> None:
    run_dir = _run_tiny_experiment(tmp_path)

    result = generate_findings(run_dir)

    assert Path(result["findings_path"]).exists()
    assert Path(result["leaderboard_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert Path(result["viewer_path"]).exists()
    findings = Path(result["findings_path"]).read_text(encoding="utf-8")
    assert "Capability Decomposition" in findings
    assert "net 95% CI" in findings
    assert "95% CI" in findings
    assert "## Model Notes" in findings
    assert "largest measured loss" in findings
    assert "## Balance Curves" in findings
    assert "final balance mean" in findings
    assert "days until insolvent" in findings
    assert "horizon active" in findings
    assert "efficiency" in findings
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert summary["schema_version"] == "solvent_findings_v0_5"
    assert summary["leaderboard"][0]["completed_cells"] == 1
    assert summary["leaderboard"][0]["net_revenue"]["min"] is not None
    assert "ci95_low" in summary["leaderboard"][0]["net_revenue"]
    assert summary["leaderboard"][0]["days_until_insolvent"]["mean"] is not None
    assert summary["leaderboard"][0]["horizon_fraction_active"]["mean"] is not None
    viewer_dir = Path(result["viewer_path"]).parent
    app_js = (viewer_dir / "app.js").read_text(encoding="utf-8")
    index_html = (viewer_dir / "index.html").read_text(encoding="utf-8")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runs"][0]["cell_id"] == summary["money_shots"]["best_efficiency"]
    assert 'id="money-shots"' in index_html
    assert "row.days_until_insolvent" in app_js
    assert "includeMin: true" in app_js
    assert "95% CI" in app_js
    assert "leaderboardColumns" in app_js
    assert "setLeaderboardSort" in app_js
    assert "data-sort" in app_js
    assert "row.selection_regret" in app_js
    assert "row.pricing_regret" in app_js
    assert "row.coherence_penalty" in app_js
    assert "combinedBalanceSvg" in app_js
    assert "combined-balance-chart" in app_js
    assert "renderMoneyShots" in app_js
    assert "runForCellId" in app_js
    assert "selectRunByKey" in app_js
    assert "curve-button" in app_js
    style_css = (viewer_dir / "style.css").read_text(encoding="utf-8")
    assert ".combined-curve" in style_css
    assert ".curve-legend" in style_css
    assert ".money-shot-card" in style_css


def test_cli_findings_generates_artifacts(tmp_path: Path) -> None:
    run_dir = _run_tiny_experiment(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "solvent.cli.main", "findings", str(run_dir), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert Path(payload["findings_path"]).exists()
    assert Path(payload["viewer_path"]).exists()


def test_cli_findings_reaggregation_is_free_without_api_credentials(tmp_path: Path) -> None:
    run_dir = _run_tiny_experiment(tmp_path)
    _relabel_first_cell(run_dir, model="gpt-5.4-mini", config_id="gpt-5.4-mini:base")

    result = subprocess.run(
        [sys.executable, "-m", "solvent.cli.main", "findings", str(run_dir), "--json"],
        check=False,
        capture_output=True,
        text=True,
        env={"SOLVENT_DOTENV": "0"},
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["leaderboard"][0]["config_id"] == "gpt-5.4-mini:base"
    assert Path(payload["findings_path"]).exists()


def _run_tiny_experiment(tmp_path: Path, conditions: str = "[redteam_off]") -> Path:
    config_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: findings-smoke",
                "models: [fake:base]",
                "seeds: [1]",
                "samples_per_seed: 1",
                f"conditions: {conditions}",
                "horizon_minutes: 60",
                "market: { arrival_rate_per_day: 24.0, decoy_rate: 0.0 }",
                "budget_usd: 1",
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
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return run_dir


def _relabel_first_cell(run_dir: Path, *, model: str, config_id: str) -> None:
    ledger_path = run_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    cell = ledger["cells"][0]
    cell["cell"]["model"] = config_id
    cell["cell"]["config_id"] = config_id
    cell["provenance"]["model"] = model
    cell["provenance"]["model_config"] = config_id
    cell["provenance"]["config_id"] = config_id
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")

    scorecard_path = Path(cell["scorecard_path"])
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    scorecard["config_id"] = config_id
    scorecard_path.write_text(json.dumps(scorecard, sort_keys=True) + "\n", encoding="utf-8")
