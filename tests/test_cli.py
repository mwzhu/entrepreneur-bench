import subprocess
import sys
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
