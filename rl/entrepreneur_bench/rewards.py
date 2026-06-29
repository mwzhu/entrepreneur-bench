from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent.scoring.reward_context import RewardContext, pricing_regret_over, selection_regret_over
from solvent.scoring.scorecard import build_reward_context

EXPECTED_NET_SCALE = Decimal("1000")
REGRET_SCALE = Decimal("1000")
SOLVENCY_PENALTY = Decimal("-1")
REWARD_COMPONENT_NAMES = (
    "r_expected_net",
    "r_pricing_neg_regret",
    "r_tool_neg_regret",
    "r_selection_neg_regret",
    "r_solvency",
)


def reward_context_from_state(state: dict[str, Any]) -> RewardContext:
    context = state.get("reward_context")
    if isinstance(context, RewardContext):
        return context
    trace_path = state.get("trace_path")
    if trace_path is None and "env" in state:
        trace_path = state["env"].config.trace_path
    if trace_path is None:
        raise ValueError("reward context requires state['reward_context'] or a trace path")
    return build_reward_context(Path(trace_path))


def reward_components(context: RewardContext) -> dict[str, float]:
    pricing_regret = pricing_regret_over(
        context.delivered_job_ids,
        context.accepted_facts,
        context.jobs_by_id,
        context.good_ids,
    )
    selection_regret = context.delivered_selection_regret
    expected_net = context.expected_net_revenue / EXPECTED_NET_SCALE
    return {
        "r_expected_net": float(expected_net),
        "r_pricing_neg_regret": -float(pricing_regret / REGRET_SCALE),
        "r_tool_neg_regret": -float(context.oracle_tool_regret / REGRET_SCALE),
        "r_selection_neg_regret": -float(selection_regret / REGRET_SCALE),
        "r_solvency": float(SOLVENCY_PENALTY if context.terminated_reason == "insolvent" else Decimal("0")),
    }


def total_reward(context: RewardContext) -> float:
    parts = reward_components(context)
    return (
        parts["r_expected_net"]
        + 0.15 * parts["r_pricing_neg_regret"]
        + 0.15 * parts["r_tool_neg_regret"]
        + 0.15 * parts["r_selection_neg_regret"]
        + 0.10 * parts["r_solvency"]
    )


def build_rubric():
    import verifiers as vf

    def terminal_reward(state: dict[str, Any], **_: Any) -> float:
        return total_reward(reward_context_from_state(state))

    rubric = vf.Rubric()
    rubric.add_reward_func(terminal_reward, weight=1.0)
    for name in REWARD_COMPONENT_NAMES:
        rubric.add_metric(_component_metric(name), weight=0.0)
    return rubric


def _component_metric(component_name: str):
    def metric(state: dict[str, Any], **_: Any) -> float:
        return reward_components(reward_context_from_state(state))[component_name]

    metric.__name__ = component_name
    return metric
