import subprocess
import sys
import json
from pathlib import Path


def test_cli_smoke_run(tmp_path: Path) -> None:
    trace_path = tmp_path / "cli.jsonl"
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
            "--horizon",
            "3",
            "--trace-path",
            str(trace_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "episode complete" in result.stdout.lower()
    trace = trace_path.read_text(encoding="utf-8")
    assert "episode_started" in trace
    assert "terminated" in trace


def test_cli_run_accepts_config_id_agent(tmp_path: Path) -> None:
    trace_path = tmp_path / "cli-config-id.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "run",
            "--agent",
            "stub:happy_path",
            "--seed",
            "42",
            "--horizon",
            "3",
            "--trace-path",
            str(trace_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "agent: stub:happy_path" in result.stdout


def test_cli_characterize_validate_menu_json() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "characterize",
            "--validate-menu",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert '"valid": true' in result.stdout


def test_cli_doctor_passes_for_fake_agent() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "doctor",
            "--agent",
            "fake:base",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["ok"] is True
    assert any(check["name"] == "recorded_replay" and check["ok"] for check in payload["checks"])


def test_cli_doctor_reports_missing_anthropic_key() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "doctor",
            "--agent",
            "claude-opus-4-8:base",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={"SOLVENT_DOTENV": "0"},
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ok"] is False
    assert any(check["name"] == "live_model_credentials" and not check["ok"] for check in payload["checks"])


def test_cli_doctor_reports_model_alias_without_secret() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "doctor",
            "--agent",
            "claude-opus-4-8:base",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            "ANTHROPIC_API_KEY": "not-a-real-secret",
            "SOLVENT_DOTENV": "0",
            "SOLVENT_MODEL_ALIAS_CLAUDE_OPUS_4_8": "claude-provider-model-id",
        },
    )
    payload = json.loads(result.stdout)
    credential = next(check for check in payload["checks"] if check["name"] == "live_model_credentials")
    assert result.returncode == 0
    assert credential["ok"] is True
    assert "claude-provider-model-id" in credential["detail"]
    assert "not-a-real-secret" not in result.stdout


def test_cli_doctor_config_passes_for_fake_experiment(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "name: doctor-fake",
                "models: [fake:base]",
                "seeds: [1]",
                "horizon_minutes: 60",
                "budget_usd: 1",
                "caching: true",
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
            "doctor",
            "--config",
            str(config),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["ok"] is True
    assert payload["name"] == "doctor-fake"
    assert payload["estimate"]["over_budget"] is False
    assert any(check["name"] == "experiment_caching" and check["ok"] for check in payload["checks"])


def test_cli_doctor_config_reports_missing_provider_key(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "name: doctor-real",
                "models: [gpt-5.4-mini:base]",
                "seeds: [1]",
                "horizon_minutes: 60",
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
            "doctor",
            "--config",
            str(config),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={"SOLVENT_DOTENV": "0"},
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["ok"] is False
    assert any(check["name"] == "brain_pricing" and check["ok"] for check in payload["checks"])
    assert any("OPENAI_API_KEY missing" in check["detail"] for check in payload["checks"])


def test_cli_doctor_config_accepts_openrouter_for_compatible_provider(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "name: doctor-openrouter",
                "models: [kimi-k2.6:base]",
                "seeds: [1]",
                "horizon_minutes: 60",
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
            "doctor",
            "--config",
            str(config),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={"SOLVENT_DOTENV": "0", "OPENROUTER_API_KEY": "not-a-real-secret"},
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["ok"] is True
    assert any("OPENROUTER_API_KEY present via OpenRouter model moonshotai/kimi-k2.6" in check["detail"] for check in payload["checks"])
    assert "not-a-real-secret" not in result.stdout


def test_cli_doctor_probe_live_reports_openrouter_chat_failure_without_secret(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-secret")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "name: doctor-openrouter-probe",
                "models: [kimi-k2.6:base]",
                "seeds: [1]",
                "horizon_minutes: 60",
                "budget_usd: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_probe(model):
        return False, "provider API error 401: User not found"

    monkeypatch.setattr("solvent.doctor.probe_openrouter_chat", fake_probe)
    from solvent.doctor import experiment_doctor

    report = experiment_doctor(config, probe_live=True)

    assert report["ok"] is False
    credential = next(check for check in report["checks"] if check["name"] == "live_model_credentials")
    assert credential["ok"] is False
    assert "User not found" in credential["detail"]
    assert "not-a-real-secret" not in json.dumps(report)


def test_cli_run_replays_recorded_sidecar_without_live_client(tmp_path: Path) -> None:
    sidecar = tmp_path / "recorded.llm.jsonl"
    sidecar.write_text(
        "\n".join(
            [
                json.dumps({"response": {"tool_call": {"name": "list_jobs", "arguments": {}}, "usage": {"input_tokens": 3, "output_tokens": 2}}}),
                json.dumps({"response": {"tool_call": {"name": "end_tick", "arguments": {}}, "usage": {"input_tokens": 4, "output_tokens": 1}}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "replay.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "run",
            "--agent",
            "fake:base",
            "--recorded-sidecar",
            str(sidecar),
            "--horizon",
            "1",
            "--temperature",
            "0.6",
            "--trace-path",
            str(trace_path),
            "--scorecard",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "brain compute cost" in result.stdout
    assert "brain_metered" in trace_path.read_text(encoding="utf-8")
    sidecar_rows = [json.loads(line) for line in trace_path.with_suffix(".llm.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sidecar_rows[0]["request"]["temperature"] == 0.6


def test_cli_run_model_max_turns_bounds_recorded_replay(tmp_path: Path) -> None:
    sidecar = tmp_path / "recorded.llm.jsonl"
    sidecar.write_text(
        "\n".join(
            [
                json.dumps({"response": {"tool_call": {"name": "list_jobs", "arguments": {}}, "usage": {"input_tokens": 3, "output_tokens": 2}}}),
                json.dumps({"response": {"tool_call": {"name": "end_tick", "arguments": {}}, "usage": {"input_tokens": 4, "output_tokens": 1}}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "bounded-replay.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "run",
            "--agent",
            "fake:base",
            "--recorded-sidecar",
            str(sidecar),
            "--model-max-turns",
            "1",
            "--model-max-tokens",
            "128",
            "--trace-path",
            str(trace_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    sidecar_rows = [json.loads(line) for line in trace_path.with_suffix(".llm.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(sidecar_rows) == 1
    assert sidecar_rows[0]["response"]["tool_call"]["name"] == "list_jobs"


def test_cli_run_missing_recorded_sidecar_is_clean_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.llm.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "run",
            "--agent",
            "fake:base",
            "--recorded-sidecar",
            str(missing),
            "--trace-path",
            str(tmp_path / "trace.jsonl"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "recorded sidecar not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_run_rejects_invalid_job_ttl_cleanly(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "run",
            "--agent",
            "stub:happy_path",
            "--job-ttl-ticks",
            "0",
            "--trace-path",
            str(tmp_path / "trace.jsonl"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "job_ttl_ticks must be at least 1" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_run_invalid_model_config_is_clean_error(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "run",
            "--agent",
            "fake:base+procedure",
            "--trace-path",
            str(tmp_path / "trace.jsonl"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid ablation spec" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_replay_scores_saved_trace_and_writes_artifacts(tmp_path: Path) -> None:
    sidecar = tmp_path / "recorded.llm.jsonl"
    sidecar.write_text(
        "\n".join(
            [
                json.dumps({"response": {"tool_call": {"name": "list_jobs", "arguments": {}}, "usage": {"input_tokens": 3, "output_tokens": 2}}}),
                json.dumps({"response": {"tool_call": {"name": "end_tick", "arguments": {}}, "usage": {"input_tokens": 4, "output_tokens": 1}}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace.jsonl"
    run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "run",
            "--agent",
            "fake:base",
            "--recorded-sidecar",
            str(sidecar),
            "--horizon",
            "1",
            "--trace-path",
            str(trace_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0

    scorecard_path = tmp_path / "scorecard.json"
    view_path = tmp_path / "view.json"
    replay_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvent.cli.main",
            "replay",
            str(trace_path),
            "--scorecard-output",
            str(scorecard_path),
            "--view-output",
            str(view_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert replay_result.returncode == 0
    assert "Solvent replay" in replay_result.stdout
    assert json.loads(scorecard_path.read_text(encoding="utf-8"))["compute"]["brain_tokens_in"] == 7
    view = json.loads(view_path.read_text(encoding="utf-8"))
    # Brain-metering is hidden from the human timeline but its cost survives in the scorecard.
    assert all(event["title"] != "Brain compute metered" for event in view["events"])
