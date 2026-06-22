import json
from decimal import Decimal
from pathlib import Path

from solvent.env.env import Environment
from solvent.env.models import EnvConfig
from solvent.env.tool_api import ToolAdapter


def test_tool_adapter_exposes_v0_4_action_space_and_public_observation(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "adapter.jsonl"))
    api = ToolAdapter(env)
    try:
        assert "clarify" in api.schemas()
        # Direct mode advertises submit but not the tool-mediated delivery tools.
        assert "submit" in api.schemas()
        assert "deliver" not in api.schemas()
        assert "list_models" not in api.schemas()
        assert api.schemas()["bid"]["input_schema"]["properties"]["price"]["type"] == "string"

        observation = api.observe()
        serialized = json.dumps(observation, sort_keys=True)
        assert "balance" in observation
        assert observation["available_jobs"]
        assert "starting_price" in observation["available_jobs"][0]
        assert "awaiting_decision" in observation
        assert observation["awaiting_decision"] == []
        assert observation["delivery_models"] == []
        assert "reservation_price" not in serialized
        assert "internal_difficulty" not in serialized
        assert "pass_prob" not in serialized
    finally:
        env.finalize()


def test_tool_mediated_observation_includes_public_delivery_menu(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "tool-mode.jsonl", delivery_mode="tool_mediated"))
    api = ToolAdapter(env)
    try:
        observation = api.observe()
        serialized = json.dumps(observation, sort_keys=True)
        assert observation["delivery_models"]
        assert {"name", "price", "capability_proxy", "speed_proxy"} <= set(observation["delivery_models"][0])
        assert "pass_prob" not in serialized
        assert "easy" not in serialized
        assert "hard" not in serialized
    finally:
        env.finalize()


def test_schemas_are_scoped_to_delivery_mode(tmp_path: Path) -> None:
    direct = ToolAdapter(Environment(_config(tmp_path / "direct.jsonl", delivery_mode="direct")))
    tool = ToolAdapter(Environment(_config(tmp_path / "tool.jsonl", delivery_mode="tool_mediated")))
    try:
        assert "submit" in direct.schemas()
        assert "deliver" not in direct.schemas()
        assert "list_models" not in direct.schemas()

        assert "submit" not in tool.schemas()
        assert "deliver" in tool.schemas()
        assert "list_models" in tool.schemas()
    finally:
        direct.env.finalize()
        tool.env.finalize()


def test_adapter_malformed_call_emits_one_invalid_action(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "malformed.jsonl"))
    api = ToolAdapter(env)
    try:
        result = api.dispatch({"name": "missing_tool", "arguments": {}})
        assert result["ok"] is False
    finally:
        summary = env.finalize()

    invalid = [event for event in _events(summary.trace_path) if event["kind"] == "invalid_action"]
    assert len(invalid) == 1
    assert invalid[0]["payload"]["code"] == "malformed_tool_call"


def test_adapter_rejects_extra_arguments_before_env_invocation(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "extra-argument.jsonl"))
    api = ToolAdapter(env)
    try:
        result = api.dispatch({"name": "list_jobs", "arguments": {"include_hidden": "true"}})
        assert result["ok"] is False
    finally:
        summary = env.finalize()

    events = _events(summary.trace_path)
    assert [event["kind"] for event in events].count("invalid_action") == 1
    assert not any(event["kind"] == "board_seen" for event in events)


def test_adapter_rejects_non_string_argument_before_env_invocation(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "typed-argument.jsonl"))
    api = ToolAdapter(env)
    try:
        job = api.dispatch({"name": "list_jobs", "arguments": {}})["result"][0]
        result = api.dispatch({"name": "bid", "arguments": {"job_id": job["id"], "price": 1.25}})
        assert result["ok"] is False
    finally:
        summary = env.finalize()

    events = _events(summary.trace_path)
    invalid = [event for event in events if event["kind"] == "invalid_action"]
    assert len(invalid) == 1
    assert invalid[0]["payload"]["code"] == "malformed_tool_call"
    assert not any(event["kind"] == "bid_made" for event in events)


def test_adapter_rejects_invalid_enum_before_env_invocation(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "enum-argument.jsonl"))
    api = ToolAdapter(env)
    try:
        result = api.dispatch(
            {"name": "respond", "arguments": {"job_id": "job_1", "decision": "negotiate"}}
        )
        assert result["ok"] is False
    finally:
        summary = env.finalize()

    invalid = [event for event in _events(summary.trace_path) if event["kind"] == "invalid_action"]
    assert len(invalid) == 1
    assert invalid[0]["payload"]["code"] == "malformed_tool_call"
    assert "negotiate" in invalid[0]["payload"]["error"]


def test_adapter_env_error_is_not_double_emitted(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "env-error.jsonl"))
    api = ToolAdapter(env)
    try:
        result = api.dispatch({"name": "inspect_job", "arguments": {"job_id": "missing"}})
        assert result["ok"] is False
    finally:
        summary = env.finalize()

    invalid = [event for event in _events(summary.trace_path) if event["kind"] == "invalid_action"]
    assert len(invalid) == 1
    assert invalid[0]["payload"]["code"] == "unknown_job"


def test_clarify_returns_public_safe_answer(tmp_path: Path) -> None:
    env = Environment(_config(tmp_path / "clarify.jsonl"))
    api = ToolAdapter(env)
    try:
        job = api.dispatch({"name": "list_jobs", "arguments": {}})["result"][0]
        result = api.dispatch({"name": "clarify", "arguments": {"job_id": job["id"], "question": "What is the budget?"}})
        assert result["ok"] is True
        assert "No additional information" in result["result"]["answer"]
    finally:
        summary = env.finalize()

    raw = summary.trace_path.read_text(encoding="utf-8")
    assert "reservation_price" not in raw
    assert "true_value" not in raw


def _config(trace_path: Path, delivery_mode: str = "direct") -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="stub:test",
        start_balance=Decimal("20.00"),
        horizon_ticks=1,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0.01"),
        trace_path=trace_path,
        delivery_mode=delivery_mode,
    )


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
