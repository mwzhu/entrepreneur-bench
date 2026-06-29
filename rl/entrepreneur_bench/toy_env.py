from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from datasets import Dataset
import verifiers as vf
from verifiers.types import Response, Tool


@dataclass
class ToyAdapter:
    calls: int = 0

    def dispatch(self, value: str) -> str:
        self.calls += 1
        return f"{value}:{self.calls}"


class OneToolToyEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs: Any):
        kwargs.setdefault("dataset", Dataset.from_list([{"question": "toy", "answer": "", "info": {}}]))
        kwargs.setdefault("rubric", vf.Rubric())
        kwargs.setdefault("parser", vf.Parser())
        super().__init__(tools=[], max_turns=1, **kwargs)
        self.tools = [toy_tool]
        self.tool_map = {"toy_tool": toy_tool}
        self.skipped_args = {"toy_tool": ["_adapter"]}
        self.tool_defs = [
            Tool(
                name="toy_tool",
                description="Dispatch a value through a hidden per-rollout adapter.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string", "description": "Value to dispatch."}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                strict=True,
            )
        ]
        self.tool_monitor_rubric.add_tool_metric("toy_tool")
        self.metered: list[dict[str, Any]] = []

    async def setup_state(self, state: vf.State) -> None:
        state["adapter"] = ToyAdapter()
        state["usage_in"] = 0
        state["usage_out"] = 0
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

    async def add_model_response(self, state: vf.State, prompt_messages: vf.Messages, response: Response):
        await super().add_model_response(state, prompt_messages, response)
        if response.usage is None:
            return
        state["usage_in"] += int(response.usage.prompt_tokens or 0)
        state["usage_out"] += int(response.usage.completion_tokens or 0) + int(response.usage.reasoning_tokens or 0)
        self.metered.append(
            {
                "cumulative_input_tokens": state["usage_in"],
                "cumulative_output_tokens": state["usage_out"],
                "cost": Decimal("0"),
            }
        )


def toy_tool(_adapter: ToyAdapter, **arguments: Any) -> str:
    return _adapter.dispatch(str(arguments["value"]))
