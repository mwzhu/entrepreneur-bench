from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

vf = pytest.importorskip("verifiers")

from verifiers.clients.client import Client
from verifiers.types import ClientConfig, Messages, Response, ResponseMessage, SamplingArgs, Tool, ToolCall, Usage

from entrepreneur_bench.environment import EntrepreneurEnv
from entrepreneur_bench.rewards import REWARD_COMPONENT_NAMES, build_rubric, reward_components
from entrepreneur_bench.seeds import build_seed_dataset
from entrepreneur_bench.toy_env import OneToolToyEnv
from solvent.env.env import Environment
from solvent.env.models import Job
from solvent.env.models import EnvConfig
from solvent.env.tool_api import ToolAdapter
from solvent.scoring.reward_context import RewardContext
from solvent.scoring.reward_context import pricing_regret_over
from solvent.scoring.scorecard import build_reward_context


def test_verifiers_load_environment_finds_package_entrypoint(tmp_path: Path) -> None:
    env = vf.load_environment("entrepreneur_bench", horizon_days=1, trace_dir=tmp_path)

    assert isinstance(env, EntrepreneurEnv)
    assert env.env_id == "entrepreneur_bench"
    assert "deliver" in {tool.name for tool in env.tool_defs}


def test_seed_dataset_carries_split_and_seed_info() -> None:
    dataset = build_seed_dataset("train", horizon_days=2)

    assert len(dataset) == 128
    assert dataset[0]["info"]["seed"] == 1000
    assert dataset[0]["info"]["split"] == "train"
    assert dataset[0]["info"]["horizon_days"] == 2


def test_delivery_gated_pricing_uses_tool_mediated_delivery_attempts(tmp_path: Path) -> None:
    delivered_summary, delivered_job_id = _tool_delivery_trace(tmp_path / "delivered.jsonl", deliver=True)
    dropped_summary, dropped_job_id = _tool_delivery_trace(tmp_path / "dropped.jsonl", deliver=False)

    delivered_context = build_reward_context(delivered_summary.trace_path)
    dropped_context = build_reward_context(dropped_summary.trace_path)

    assert delivered_context.delivered_job_ids == {delivered_job_id}
    assert dropped_context.delivered_job_ids == set()
    assert pricing_regret_over(
        dropped_context.delivered_job_ids,
        dropped_context.accepted_facts,
        dropped_context.jobs_by_id,
        dropped_context.good_ids,
    ) == Decimal("0.00")
    assert dropped_job_id in dropped_context.accepted_facts
    assert reward_components(dropped_context)["r_pricing_neg_regret"] == 0.0


def test_rubric_surfaces_zero_weight_reward_component_metrics(tmp_path: Path) -> None:
    summary, _ = _tool_delivery_trace(tmp_path / "rubric-components.jsonl", deliver=True)
    context = build_reward_context(summary.trace_path)
    rubric = build_rubric()
    state = {"reward_context": context, "prompt": [], "completion": []}

    asyncio.run(rubric.score_rollout(state))

    assert set(REWARD_COMPONENT_NAMES) < set(state["metrics"])
    assert state["metrics"]["terminal_reward"] == state["reward"]
    for name in REWARD_COMPONENT_NAMES:
        assert state["metrics"][name] == reward_components(context)[name]


def test_reward_shaping_is_dominated_by_terminal_profit_across_train_seeds(tmp_path: Path) -> None:
    failures = []
    for row in build_seed_dataset("train", horizon_days=2):
        seed = int(row["info"]["seed"])
        noop = _run_train_seed_policy(tmp_path / f"{seed}-noop.jsonl", seed, near_optimal=False)
        near = _run_train_seed_policy(tmp_path / f"{seed}-near.jsonl", seed, near_optimal=True)
        noop_context = build_reward_context(noop.trace_path)
        near_context = build_reward_context(near.trace_path)

        terminal_delta = _terminal_reward_component(near_context) - _terminal_reward_component(noop_context)
        shaping_envelope = max(_weighted_shaping_envelope(noop_context), _weighted_shaping_envelope(near_context))
        if not shaping_envelope < terminal_delta:
            failures.append((seed, terminal_delta, shaping_envelope))

    assert failures == []


def test_entrepreneur_env_advertises_mode_gated_schemas_without_hidden_adapter(tmp_path: Path) -> None:
    env = EntrepreneurEnv(trace_dir=tmp_path)
    tool_defs = {tool.name: tool for tool in env.tool_defs}

    assert "deliver" in tool_defs
    assert "list_models" in tool_defs
    assert "submit" not in tool_defs
    for tool in tool_defs.values():
        assert "_adapter" not in tool.parameters.get("properties", {})


def test_one_tool_toy_env_proves_hidden_arg_and_usage_contract() -> None:
    env = OneToolToyEnv()
    state = {"trajectory_id": "toy", "trajectory": []}
    asyncio.run(env.setup_state(state))

    tool_def = env.tool_defs[0]
    assert tool_def.name == "toy_tool"
    assert "_adapter" not in tool_def.parameters.get("properties", {})
    assert env.skipped_args["toy_tool"] == ["_adapter"]

    injected = env.update_tool_args("toy_tool", {"value": "ping"}, [], state)
    assert "_adapter" in injected
    assert env.tool_map["toy_tool"](**injected) == "ping:1"

    asyncio.run(env.add_model_response(state, [], _response(prompt_tokens=4, completion_tokens=3, reasoning_tokens=2)))
    asyncio.run(env.add_model_response(state, [], _response(prompt_tokens=6, completion_tokens=1, reasoning_tokens=0)))

    assert env.metered[-1]["cumulative_input_tokens"] == 10
    assert env.metered[-1]["cumulative_output_tokens"] == 6
    assert env.metered[-1]["cost"] == Decimal("0")


def test_policy_usage_metering_emits_brain_metered_trace_rows(tmp_path: Path) -> None:
    env = EntrepreneurEnv(trace_dir=tmp_path)
    state = {
        "trajectory_id": "meter",
        "info": {"seed": 1000, "config_id": "rl:test", "split": "train", "horizon_days": 1},
    }
    asyncio.run(env.setup_state(state))

    env._record_compute(state, _response(prompt_tokens=11, completion_tokens=7, reasoning_tokens=3))
    env._record_compute(state, _response(prompt_tokens=5, completion_tokens=2, reasoning_tokens=0))
    summary = state["env"].finalize()

    events = _events(summary.trace_path)
    metered = [event for event in events if event["kind"] == "brain_metered"]
    assert len(metered) == 2
    assert metered[-1]["payload"]["cumulative_input_tokens"] == 16
    assert metered[-1]["payload"]["cumulative_output_tokens"] == 12
    assert metered[-1]["payload"]["cumulative_cost"] == "0.000000"


def test_scripted_wrapper_rollout_finalizes_scores_and_records_token_canary(tmp_path: Path) -> None:
    env = EntrepreneurEnv(trace_dir=tmp_path)
    state = {
        "trajectory_id": "scripted",
        "info": {"seed": 1000, "config_id": "rl:test", "split": "train", "horizon_days": "1"},
    }
    asyncio.run(env.setup_state(state))

    env._record_compute(state, _response(prompt_tokens=13, completion_tokens=5, reasoning_tokens=2))
    listed = _call_wrapper_tool(env, state, "list_jobs")
    for _ in range(10):
        if listed["result"]["result"] or state["env"].terminated():
            break
        _call_wrapper_tool(env, state, "advance_to_next_event")
        listed = _call_wrapper_tool(env, state, "list_jobs")

    job = listed["result"]["result"][0]
    accepted = _call_wrapper_tool(env, state, "accept", job_id=job["id"])
    if accepted["result"]["result"].get("manipulation") is not None:
        _call_wrapper_tool(env, state, "respond", job_id=job["id"], decision="hold_price")

    models = _call_wrapper_tool(env, state, "list_models")["result"]["result"]
    _call_wrapper_tool(env, state, "deliver", job_id=job["id"], model=models[0]["name"])

    asyncio.run(env.finalize_solvent(state))
    asyncio.run(env.finalize_solvent(state))

    assert state["summary"].trace_path == Path(state["trace_path"])
    assert state["reward_context"].trace_path == Path(state["trace_path"])
    assert state["scorecard"].compute.brain_tokens_in == 13
    assert state["scorecard"].compute.brain_tokens_out == 7
    token_canary = state["scorecard"].expected_net_revenue / max(state["scorecard"].compute.brain_tokens_out, 1)
    assert token_canary.is_finite()


def test_verifiers_rollout_loop_executes_tool_calls_and_scores(tmp_path: Path) -> None:
    env = EntrepreneurEnv(horizon_days=1, trace_dir=tmp_path, max_turns=40)
    rollout_input = env.get_dataset(n=1).to_list()[0]

    state = asyncio.run(env.rollout(rollout_input, _ScriptedToolClient(), "scripted-policy", {"temperature": 0}))
    asyncio.run(env.rubric.score_rollout(state))

    assert state["error"] is None
    assert state["summary"].trace_path == Path(state["trace_path"])
    assert state["scorecard"].trace_path == Path(state["trace_path"])
    assert state["reward"] is not None
    assert state["scorecard"].compute.brain_tokens_in > 0
    assert state["scorecard"].compute.brain_tokens_out > 0
    assert any(step["completion"][0].tool_calls for step in state["trajectory"] if step["completion"])
    assert Path(state["trace_path"]).exists()


def _tool_delivery_trace(trace_path: Path, *, deliver: bool):
    env = Environment(
        EnvConfig(
            seed=42,
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
        )
    )
    api = ToolAdapter(env)
    try:
        listed = api.dispatch({"name": "list_jobs", "arguments": {}})
        job = listed["result"][0]
        api.dispatch({"name": "bid", "arguments": {"job_id": job["id"], "price": "0.50"}})
        if deliver:
            api.dispatch({"name": "deliver", "arguments": {"job_id": job["id"], "model": "tool-pro"}})
    finally:
        summary = env.finalize()
    return summary, job["id"]


def _run_train_seed_policy(trace_path: Path, seed: int, *, near_optimal: bool):
    env = Environment(_rl_train_config(trace_path, seed))
    try:
        guard = 0
        while not env.terminated() and guard < 256:
            guard += 1
            acted = False
            if near_optimal:
                for public in list(env.available_jobs()):
                    if public.id in env.accepted_jobs or public.id in env.declined_jobs:
                        continue
                    job = env.market.get_job(public.id)
                    model_name, expected_value = _best_delivery_model(env, job)
                    if not job.is_decoy and expected_value > 0:
                        env.accept(public.id)
                        if public.id in env.pending_manipulations:
                            env.respond(public.id, "hold_price")
                        env.deliver(public.id, model_name)
                    else:
                        env.decline(public.id)
                    acted = True
                    if env.terminated():
                        break
            if not acted and not env.terminated():
                env.advance_to_next_event()
    finally:
        return env.finalize()


def _rl_train_config(trace_path: Path, seed: int) -> EnvConfig:
    horizon_minutes = 2 * 1440
    return EnvConfig(
        seed=seed,
        config_id="rl:test",
        start_balance=Decimal("1000.00"),
        horizon_ticks=horizon_minutes,
        horizon_minutes=horizon_minutes,
        overhead_per_tick=Decimal("0"),
        overhead_per_minute=Decimal("0.006944"),
        tool_call_cost=Decimal("0"),
        trace_path=trace_path,
        market_version="business_stream_v0_5",
        market_size=max(1, round(Decimal("4.50") * Decimal(horizon_minutes) / Decimal("1440"))),
        arrival_rate_per_day=Decimal("4.50"),
        decoy_rate=Decimal("0.40"),
        delivery_mode="tool_mediated",
        task_mix={"data_clean": 1.0},
        difficulty_distribution={"easy": 1.0},
        job_ttl_minutes=1440,
        seed_split="train",
        brain_model="qwen3-4b-instruct",
        breach_fee_frac=Decimal("0.25"),
    )


def _best_delivery_model(env: Environment, job: Job) -> tuple[str, Decimal]:
    best_name = ""
    best_value = Decimal("-Infinity")
    for model in env.delivery_menu.public_models():
        pass_prob = Decimal(str(env.delivery_menu.pass_prob(job.type, model.name, job.internal_difficulty)))
        expected_value = job.reservation_price * pass_prob - model.price
        if expected_value > best_value:
            best_name = model.name
            best_value = expected_value
    return best_name, best_value


def _terminal_reward_component(context: RewardContext) -> Decimal:
    return Decimal(str(reward_components(context)["r_expected_net"]))


def _weighted_shaping_envelope(context: RewardContext) -> Decimal:
    parts = reward_components(context)
    return (
        Decimal("0.15")
        * (
            abs(Decimal(str(parts["r_pricing_neg_regret"])))
            + abs(Decimal(str(parts["r_tool_neg_regret"])))
            + abs(Decimal(str(parts["r_selection_neg_regret"])))
        )
        + Decimal("0.10") * abs(Decimal(str(parts["r_solvency"])))
    )


def _response(prompt_tokens: int, completion_tokens: int, reasoning_tokens: int) -> Response:
    return Response(
        id="response",
        created=0,
        model="test",
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=prompt_tokens + completion_tokens + reasoning_tokens,
        ),
        message=ResponseMessage(
            role="assistant",
            content="",
            tool_calls=None,
            finish_reason="stop",
            is_truncated=False,
        ),
    )


def _events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]


def _call_wrapper_tool(env: EntrepreneurEnv, state: dict, name: str, **arguments) -> dict:
    injected = env.update_tool_args(name, arguments, [], state)
    return json.loads(env.tool_map[name](**injected))


class _ScriptedToolClient(Client[None, Messages, Response, Tool]):
    def __init__(self) -> None:
        super().__init__(None)
        self.calls = 0

    def setup_client(self, config: ClientConfig) -> None:
        return None

    async def to_native_tool(self, tool: Tool) -> Tool:
        return tool

    async def to_native_prompt(self, messages: Messages) -> tuple[Messages, dict]:
        return messages, {}

    async def get_native_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Response:
        state = kwargs["state"]
        tool_call = self._next_tool_call(state)
        self.calls += 1
        return Response(
            id=f"scripted-{self.calls}",
            created=self.calls,
            model=model,
            usage=Usage(prompt_tokens=5, completion_tokens=3, reasoning_tokens=0, total_tokens=8),
            message=ResponseMessage(
                role="assistant",
                content=None if tool_call is not None else "done",
                tool_calls=[tool_call] if tool_call is not None else None,
                finish_reason="tool_calls" if tool_call is not None else "stop",
                is_truncated=False,
            ),
        )

    async def raise_from_native_response(self, response: Response) -> None:
        return None

    async def from_native_response(self, response: Response) -> Response:
        return response

    async def close(self) -> None:
        return None

    def _next_tool_call(self, state: dict[str, Any]) -> ToolCall | None:
        env = state["env"]
        if env.terminated():
            return None
        available = list(env.available_jobs())
        if not available and not env.accepted_jobs:
            return _tool_call(self.calls, "advance_to_next_event", {})
        if not env.accepted_jobs:
            return _tool_call(self.calls, "accept", {"job_id": available[0].id})
        if not state.get("listed_models"):
            state["listed_models"] = True
            return _tool_call(self.calls, "list_models", {})
        in_progress = [job_id for job_id, accepted in env.accepted_jobs.items() if not accepted.submitted]
        if in_progress:
            model = env.delivery_menu.public_models()[0].name
            return _tool_call(self.calls, "deliver", {"job_id": in_progress[0], "model": model})
        return _tool_call(self.calls, "advance_to_next_event", {})


def _tool_call(index: int, name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(id=f"call-{index}", name=name, arguments=json.dumps(arguments, sort_keys=True))
