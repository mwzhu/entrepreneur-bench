import json
from decimal import Decimal
from pathlib import Path

from solvent.cli.main import run_episode
from solvent.env.models import EnvConfig
from solvent.harness.context import ContextManager, estimate_tokens
from solvent.harness.llm import LLMHarness
from solvent.harness.model_client import FakeClient, ModelResponse


def test_sliding_window_caps_history_by_estimated_tokens() -> None:
    history = [
        {"turn": index, "payload": "x" * 80}
        for index in range(12)
    ]
    context = ContextManager("sliding_window", window_tokens=80).build(history)

    assert context
    assert estimate_tokens(context) <= 100
    assert context[-1]["turn"] == 11
    assert context[0]["turn"] > 0


def test_scratchpad_collapses_history_to_summary() -> None:
    history = [{"tool_call": {"name": "list_jobs"}, "result": {"ok": True}} for _ in range(20)]

    context = ContextManager("scratchpad", window_tokens=40).build(history)

    assert len(context) == 1
    assert context[0]["scratchpad"]["turns_seen"] == 20


def test_llm_sidecar_records_bounded_context_not_full_history(tmp_path: Path) -> None:
    responses = [ModelResponse({"name": "list_jobs", "arguments": {}}) for _ in range(8)]
    responses.append(ModelResponse({"name": "end_tick", "arguments": {}}))
    trace_path = tmp_path / "context-sidecar.jsonl"

    run_episode(
        _config(trace_path),
        LLMHarness(
            model="fake",
            client=FakeClient(responses),
            max_turns=9,
            context_policy="sliding_window",
            ctx_window_tokens=30,
        ),
    )

    rows = [json.loads(line) for line in trace_path.with_suffix(".llm.jsonl").read_text(encoding="utf-8").splitlines()]

    assert rows[-1]["request"]["history_turns_seen"] > len(rows[-1]["request"]["context_history"])
    assert "history" not in rows[-1]["request"]
    assert rows[-1]["request"]["context_policy"] == "sliding_window"


def test_llm_request_payload_reserves_room_for_system_tools_and_observation(tmp_path: Path) -> None:
    responses = [ModelResponse({"name": "list_jobs", "arguments": {}}) for _ in range(10)]
    responses.append(ModelResponse({"name": "end_tick", "arguments": {}}))
    client = FakeClient(responses)
    trace_path = tmp_path / "context-budget.jsonl"
    window = 1700

    run_episode(
        _config(trace_path),
        LLMHarness(
            model="fake",
            client=client,
            max_turns=11,
            context_policy="sliding_window",
            ctx_window_tokens=window,
        ),
    )

    assert len(client.requests) == 11
    for request in client.requests:
        payload_tokens = estimate_tokens(
            {
                "system_prompt": request.system_prompt,
                "tools": request.tools,
                "observation": request.observation,
                "history": request.history,
            }
        )
        assert payload_tokens <= window


def test_llm_sidecar_row_size_stays_bounded_over_many_turns(tmp_path: Path) -> None:
    responses = [ModelResponse({"name": "list_jobs", "arguments": {}}) for _ in range(30)]
    responses.append(ModelResponse({"name": "end_tick", "arguments": {}}))
    trace_path = tmp_path / "linear-sidecar.jsonl"

    run_episode(
        _config(trace_path),
        LLMHarness(
            model="fake",
            client=FakeClient(responses),
            max_turns=31,
            context_policy="sliding_window",
            ctx_window_tokens=1600,
        ),
    )

    lines = trace_path.with_suffix(".llm.jsonl").read_text(encoding="utf-8").splitlines()
    row_sizes = [len(line) for line in lines]

    assert len(lines) == 31
    assert max(row_sizes) < 2 * row_sizes[0]
    assert sum(row_sizes) < len(lines) * 2 * row_sizes[0]


def _config(trace_path: Path) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="fake:base",
        start_balance=Decimal("20.00"),
        horizon_ticks=1,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0"),
        trace_path=trace_path,
        delivery_mode="tool_mediated",
    )
