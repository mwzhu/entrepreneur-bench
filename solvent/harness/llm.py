from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent.env.env import Environment
from solvent.env.pricing import TokenUsage, brain_cost
from solvent.env.tool_api import ToolAdapter
from solvent.harness.context import ContextManager, estimate_tokens
from solvent.harness.model_client import ModelClient, ModelRequest, client_for_model, response_to_dict
from solvent.harness.prompts import system_prompt

ABLATIONS = {"memory", "planner", "procedure", "economic"}
REQUEST_ENVELOPE_TOKENS = 128


class BudgetExceededError(RuntimeError):
    """Raised when a single harness episode exceeds its reserved spend."""


class LLMHarness:
    """Provider-neutral ReAct harness that drives Solvent through ToolAdapter."""

    def __init__(
        self,
        model: str,
        ablations: set[str] | None = None,
        client: ModelClient | None = None,
        max_turns: int = 200,
        model_max_tokens: int = 1024,
        sidecar_path: Path | None = None,
        temperature: float = 0.0,
        context_policy: str = "sliding_window",
        ctx_window_tokens: int = 30000,
        caching: bool = True,
        budget_limit: Decimal | None = None,
        reasoning: bool = True,
    ):
        self.model = model
        self.ablations = set(ablations or set())
        unknown = self.ablations - ABLATIONS
        if unknown:
            raise ValueError(f"unknown LLM ablations: {', '.join(sorted(unknown))}")
        if temperature < 0 or temperature > 1:
            raise ValueError("temperature must be between 0 and 1")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if model_max_tokens < 1:
            raise ValueError("model_max_tokens must be at least 1")
        self.client = client or client_for_model(model, temperature=temperature, max_tokens=model_max_tokens)
        self.max_turns = max_turns
        self.model_max_tokens = model_max_tokens
        self.sidecar_path = sidecar_path
        self.temperature = temperature
        self.context = ContextManager(context_policy, ctx_window_tokens)
        self.caching = caching
        self.reasoning = reasoning
        self.usage = TokenUsage()
        self.cost = Decimal("0")
        self.budget_limit = budget_limit

    @classmethod
    def from_config_id(
        cls,
        config_id: str,
        client: ModelClient | None = None,
        temperature: float = 0.0,
        max_turns: int = 200,
        model_max_tokens: int = 1024,
        context_policy: str = "sliding_window",
        ctx_window_tokens: int = 30000,
        caching: bool = True,
        budget_limit: Decimal | None = None,
        reasoning: bool = True,
    ) -> "LLMHarness":
        model, spec = config_id.split(":", 1) if ":" in config_id else (config_id, "base")
        return cls(
            model=model,
            ablations=parse_ablation_spec(spec),
            client=client,
            temperature=temperature,
            max_turns=max_turns,
            model_max_tokens=model_max_tokens,
            context_policy=context_policy,
            ctx_window_tokens=ctx_window_tokens,
            caching=caching,
            budget_limit=budget_limit,
            reasoning=reasoning,
        )

    def run(self, env: Environment) -> None:
        api = ToolAdapter(env)
        history: list[dict[str, Any]] = []
        sidecar_path = self.sidecar_path or env.config.trace_path.with_suffix(".llm.jsonl")
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with sidecar_path.open("w", encoding="utf-8") as sidecar:
            turn = 0
            while not env.terminated() and turn < self.max_turns:
                observation = api.observe()
                prompt = system_prompt(self.ablations)
                tools = api.schemas()
                reserved_tokens = estimate_tokens(
                    {
                        "system_prompt": prompt,
                        "observation": observation,
                        "tools": tools,
                    }
                ) + REQUEST_ENVELOPE_TOKENS
                context_history = self.context.build(history, reserved_tokens=reserved_tokens)
                request = ModelRequest(
                    model=self.model,
                    system_prompt=prompt,
                    observation=observation,
                    tools=tools,
                    history=context_history,
                    temperature=self.temperature,
                    cache_hint=self.caching,
                    reasoning=self.reasoning,
                )
                response = self.client.complete(request)
                self._record_compute(env, response.usage)
                self._raise_if_budget_exceeded(env)
                result = api.dispatch(response.tool_call)
                record = {
                    "turn": turn,
                    "request": {
                        "model": request.model,
                        "system_prompt": request.system_prompt,
                        "observation": request.observation,
                        "tools": request.tools,
                        "context_policy": self.context.policy,
                        "ctx_window_tokens": self.context.window_tokens,
                        "caching": self.caching,
                        "cache_hint": request.cache_hint,
                        "context_history": request.history,
                        "history_turns_seen": turn,
                        "request_tokens_estimate": estimate_tokens(
                            {
                                "system_prompt": request.system_prompt,
                                "observation": request.observation,
                                "tools": request.tools,
                                "history": request.history,
                            }
                        ),
                        "temperature": request.temperature,
                    },
                    "response": response_to_dict(response),
                    "dispatch": result,
                }
                sidecar.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                sidecar.flush()
                history.append({"observation": observation, "tool_call": response.tool_call, "result": result})
                history = self.context.trim_memory(history)
                turn += 1

    def _record_compute(self, env: Environment, usage: TokenUsage) -> None:
        self.usage = TokenUsage(
            input_tokens=self.usage.input_tokens + usage.input_tokens,
            output_tokens=self.usage.output_tokens + usage.output_tokens,
            cache_read_tokens=self.usage.cache_read_tokens + usage.cache_read_tokens,
            cache_write_tokens=self.usage.cache_write_tokens + usage.cache_write_tokens,
        )
        cost = brain_cost(self.model, usage)
        self.cost += cost
        env._emit(
            "brain_metered",
            {
                "model": self.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "cost": cost,
                "cumulative_input_tokens": self.usage.input_tokens,
                "cumulative_output_tokens": self.usage.output_tokens,
                "cumulative_cache_read_tokens": self.usage.cache_read_tokens,
                "cumulative_cache_write_tokens": self.usage.cache_write_tokens,
                "cumulative_cost": self.cost,
                "ablations": sorted(self.ablations),
            },
            Decimal("0"),
        )

    def _raise_if_budget_exceeded(self, env: Environment) -> None:
        if self.budget_limit is None or self.cost <= self.budget_limit:
            return
        env._emit(
            "budget_exceeded",
            {
                "model": self.model,
                "budget_limit": self.budget_limit,
                "cumulative_cost": self.cost,
            },
            Decimal("0"),
        )
        raise BudgetExceededError(f"cell budget exceeded: {self.cost} > {self.budget_limit}")


def parse_ablation_spec(spec: str) -> set[str]:
    if spec in {"", "base"}:
        return set()
    parts = [part for part in spec.split("+") if part]
    if not parts or "base" in parts:
        raise ValueError(f"invalid ablation spec: {spec}")
    ablations = set(parts)
    unknown = ablations - ABLATIONS
    if unknown:
        raise ValueError(f"unknown ablations: {', '.join(sorted(unknown))}")
    return ablations
