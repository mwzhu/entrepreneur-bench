from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import verifiers as vf
from verifiers.types import Response, Tool

from entrepreneur_bench.rewards import build_rubric
from entrepreneur_bench.seeds import build_seed_dataset
from solvent.env.env import Environment
from solvent.env.models import EnvConfig
from solvent.env.pricing import TokenUsage, brain_cost
from solvent.env.tool_api import TOOL_SCHEMAS, ToolAdapter, schemas_for_delivery_mode
from solvent.harness.prompts import system_prompt
from solvent.scoring.scorecard import build_reward_context, score_trace

CANONICAL_MODEL_ID = "qwen3-4b-instruct"


def load_environment(
    horizon_days: int = 2,
    split: str = "train",
    delivery_mode: str = "tool_mediated",
    job_ttl: bool = True,
    breach_fee_frac: str | float | Decimal = Decimal("0.25"),
    brain_model: str = CANONICAL_MODEL_ID,
    **kwargs: Any,
) -> "EntrepreneurEnv":
    return EntrepreneurEnv(
        dataset=build_seed_dataset(split=split, horizon_days=horizon_days),
        rubric=build_rubric(),
        parser=vf.Parser(),
        horizon_days=horizon_days,
        split=split,
        delivery_mode=delivery_mode,
        job_ttl=job_ttl,
        breach_fee_frac=Decimal(str(breach_fee_frac)),
        brain_model=brain_model,
        **kwargs,
    )


class EntrepreneurEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        *,
        horizon_days: int = 2,
        split: str = "train",
        delivery_mode: str = "tool_mediated",
        job_ttl: bool = True,
        breach_fee_frac: Decimal = Decimal("0.25"),
        brain_model: str = CANONICAL_MODEL_ID,
        trace_dir: str | Path = "rl/artifacts/traces",
        **kwargs: Any,
    ):
        self.horizon_days = horizon_days
        self.split = split
        self.delivery_mode = delivery_mode
        self.job_ttl = job_ttl
        self.breach_fee_frac = breach_fee_frac
        self.brain_model = brain_model
        self.trace_dir = Path(trace_dir)
        max_turns = kwargs.pop("max_turns", _max_turns(horizon_days))
        kwargs.setdefault("dataset", build_seed_dataset(split=split, horizon_days=horizon_days))
        kwargs.setdefault("rubric", build_rubric())
        kwargs.setdefault("parser", vf.Parser())
        super().__init__(tools=[], max_turns=max_turns, system_prompt=system_prompt(set()), **kwargs)
        self._install_solvent_tools()

    async def setup_state(self, state: vf.State) -> None:
        info = state.get("info") or state.get("task", {}).get("info") or {}
        horizon_days = _decimal_metadata(info.get("horizon_days", self.horizon_days), "horizon_days")
        horizon_minutes = int(horizon_days * Decimal("1440"))
        seed = int(info["seed"])
        trace_path = self.trace_dir / f"{state['trajectory_id']}-seed-{seed}.jsonl"
        cfg = EnvConfig(
            seed=seed,
            config_id=str(info.get("config_id", "rl:qwen3-4b")),
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
            delivery_mode=self.delivery_mode,
            task_mix={"data_clean": 1.0},
            difficulty_distribution={"easy": 1.0},
            job_ttl_minutes=min(1440, horizon_minutes) if self.job_ttl else None,
            seed_split=str(info.get("split", self.split)),
            brain_model=self.brain_model,
            breach_fee_frac=self.breach_fee_frac,
        )
        env = Environment(cfg)
        state["env"] = env
        state["adapter"] = ToolAdapter(env)
        state["trace_path"] = str(trace_path)
        state["policy_usage"] = TokenUsage()
        state["policy_cost"] = Decimal("0")
        await super().setup_state(state)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        return {**tool_args, "_adapter": state["adapter"]}

    @vf.stop
    async def solvent_terminated(self, state: vf.State, **_: Any) -> bool:
        env = state.get("env")
        return bool(env is not None and env.terminated())

    async def add_model_response(self, state: vf.State, prompt_messages: vf.Messages, response: Response):
        await super().add_model_response(state, prompt_messages, response)
        self._record_compute(state, response)

    @vf.cleanup(priority=50)
    async def finalize_solvent(self, state: vf.State) -> None:
        env = state.get("env")
        if env is None:
            return
        summary = env.finalize()
        state["summary"] = summary
        state["scorecard"] = score_trace(summary.trace_path)
        state["reward_context"] = build_reward_context(summary.trace_path)

    def _record_compute(self, state: vf.State, response: Response) -> None:
        env = state.get("env")
        usage_raw = response.usage
        if env is None or usage_raw is None:
            return
        usage = TokenUsage(
            input_tokens=_usage_int(usage_raw.prompt_tokens),
            output_tokens=_usage_int(usage_raw.completion_tokens) + _usage_int(getattr(usage_raw, "reasoning_tokens", 0)),
        )
        current = state.get("policy_usage", TokenUsage())
        cumulative = TokenUsage(
            input_tokens=current.input_tokens + usage.input_tokens,
            output_tokens=current.output_tokens + usage.output_tokens,
            cache_read_tokens=current.cache_read_tokens + usage.cache_read_tokens,
            cache_write_tokens=current.cache_write_tokens + usage.cache_write_tokens,
        )
        cost = brain_cost(self.brain_model, usage)
        cumulative_cost = state.get("policy_cost", Decimal("0")) + cost
        state["policy_usage"] = cumulative
        state["policy_cost"] = cumulative_cost
        env._emit(
            "brain_metered",
            {
                "model": self.brain_model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "cost": cost,
                "cumulative_input_tokens": cumulative.input_tokens,
                "cumulative_output_tokens": cumulative.output_tokens,
                "cumulative_cache_read_tokens": cumulative.cache_read_tokens,
                "cumulative_cache_write_tokens": cumulative.cache_write_tokens,
                "cumulative_cost": cumulative_cost,
                "ablations": [],
            },
            Decimal("0"),
        )

    def _install_solvent_tools(self) -> None:
        tool_names = list(_schemas_for_mode(self.delivery_mode))
        tools = [_make_solvent_tool(name) for name in tool_names]
        self.tools = tools
        self.tool_map = {tool.__name__: tool for tool in tools}
        for name in tool_names:
            if hasattr(self, "tool_monitor_rubric"):
                self.tool_monitor_rubric.add_tool_metric(name)
        self.skipped_args = {name: ["_adapter"] for name in tool_names}
        self.tool_defs = [_tool_def_from_solvent_schema(name, TOOL_SCHEMAS[name]) for name in tool_names]


def _make_solvent_tool(name: str) -> Callable[..., str]:
    def _tool(_adapter: ToolAdapter, **arguments: Any) -> str:
        result = _adapter.dispatch({"name": name, "arguments": arguments})
        return json.dumps({"result": result, "observation": _adapter.observe()}, sort_keys=True, default=str)

    _tool.__name__ = name
    _tool.__doc__ = str(TOOL_SCHEMAS[name]["description"])
    return _tool


def _tool_def_from_solvent_schema(name: str, schema: dict[str, Any]) -> Tool:
    return Tool(
        name=name,
        description=str(schema["description"]),
        parameters=dict(schema["input_schema"]),
        strict=True,
    )


def _schemas_for_mode(delivery_mode: str) -> dict[str, dict[str, Any]]:
    return schemas_for_delivery_mode(delivery_mode)


def _max_turns(horizon_days: int) -> int:
    expected_jobs = max(1, round(Decimal("4.50") * Decimal(horizon_days)))
    return int(expected_jobs * 10 + 200)


def _decimal_metadata(value: Any, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc


def _usage_int(value: Any) -> int:
    return int(value or 0)
