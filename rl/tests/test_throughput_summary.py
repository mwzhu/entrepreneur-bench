from __future__ import annotations

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from solvent.env.env import Environment
from solvent.env.models import EnvConfig


def test_throughput_summary_marks_partial_until_min_episode_gate(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    trace_paths = [_trace(tmp_path / f"trace-{index}.jsonl", seed=1000 + index) for index in range(2)]
    _write_results(results_dir / "results.jsonl", trace_paths)

    partial = tmp_path / "partial.md"
    subprocess.run(
        [
            sys.executable,
            "rl/scripts/summarize_throughput.py",
            str(results_dir),
            "--hourly-rate",
            "0.27",
            "--node-spec",
            "1x A6000 spot",
            "--output",
            str(partial),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )
    partial_text = partial.read_text(encoding="utf-8")
    assert "Status: partial measurement" in partial_text
    assert "T1 not satisfied: need at least 64 clean episodes." in partial_text
    assert "Cost per 1k C1 episodes:" in partial_text

    measured = tmp_path / "measured.md"
    subprocess.run(
        [
            sys.executable,
            "rl/scripts/summarize_throughput.py",
            str(results_dir),
            "--hourly-rate",
            "0.27",
            "--node-spec",
            "1x A6000 spot",
            "--min-episodes",
            "2",
            "--output",
            str(measured),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )
    measured_text = measured.read_text(encoding="utf-8")
    assert "Status: measured" in measured_text
    assert "Gate note: T1 satisfied." in measured_text
    assert "| Trace | Turns | Wall s | Input tokens | Output tokens |" in measured_text


def _trace(trace_path: Path, *, seed: int) -> Path:
    env = Environment(
        EnvConfig(
            seed=seed,
            config_id="rl:test",
            start_balance=Decimal("1000.00"),
            horizon_ticks=5,
            overhead_per_tick=Decimal("0"),
            tool_call_cost=Decimal("0"),
            trace_path=trace_path,
            market_version="data_clean_static_v0_2",
            market_size=5,
            decoy_rate=Decimal("0.40"),
            delivery_mode="tool_mediated",
            breach_fee_frac=Decimal("0.25"),
            brain_model="qwen3-4b-instruct",
        )
    )
    try:
        job = env.list_jobs()[0]
        env.accept(job.id)
        env.deliver(job.id, env.delivery_menu.public_models()[0].name)
        env._emit(
            "brain_metered",
            {
                "model": "qwen3-4b-instruct",
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost": Decimal("0"),
                "cumulative_input_tokens": 11,
                "cumulative_output_tokens": 7,
                "cumulative_cache_read_tokens": 0,
                "cumulative_cache_write_tokens": 0,
                "cumulative_cost": Decimal("0"),
                "ablations": [],
            },
            Decimal("0"),
        )
    finally:
        return env.finalize().trace_path


def _write_results(path: Path, trace_paths: list[Path]) -> None:
    rows = []
    for index, trace_path in enumerate(trace_paths):
        rows.append(
            {
                "example_id": index,
                "trace_path": str(trace_path),
                "timing": {"total": 10.0 + index, "model": {"duration": 8.0 + index}},
                "token_usage": {"input_tokens": 20 + index, "output_tokens": 12 + index},
                "completion": [
                    {"role": "assistant", "content": None, "tool_calls": [{"name": "accept"}]},
                    {"role": "tool", "content": "{}"},
                    {"role": "assistant", "content": None, "tool_calls": [{"name": "deliver"}]},
                    {"role": "tool", "content": "{}"},
                ],
                "error": None,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
