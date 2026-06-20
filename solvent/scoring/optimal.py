from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from solvent.env.models import Job
from solvent.scoring.events import TraceFacts


@dataclass(frozen=True)
class ReferenceResult:
    net: Decimal
    selected_jobs: list[Job]


def reachable_jobs(jobs: list[Job], horizon_ticks: int) -> list[Job]:
    return [job for job in jobs if job.arrival_tick < horizon_ticks]


def omniscient_reference(jobs: list[Job], facts: TraceFacts) -> ReferenceResult:
    selected = [job for job in reachable_jobs(jobs, facts.horizon_ticks) if job.true_value > 0]
    return ReferenceResult(net=_reference_net(selected, facts, Decimal("1.0")), selected_jobs=selected)


def realizable_reference(jobs: list[Job], facts: TraceFacts, average_verify_score: float | None) -> ReferenceResult:
    capability = Decimal(str(average_verify_score)) if average_verify_score is not None else Decimal("1.0")
    selected = [
        job
        for job in reachable_jobs(jobs, facts.horizon_ticks)
        if (job.reservation_price * capability) - job.est_cost > 0
    ]
    return ReferenceResult(net=_reference_net(selected, facts, capability), selected_jobs=selected)


def _reference_net(selected: list[Job], facts: TraceFacts, capability: Decimal) -> Decimal:
    gross_profit = sum(((job.reservation_price * capability) - job.est_cost for job in selected), Decimal("0"))
    expected_overhead = Decimal(facts.horizon_ticks) * facts.overhead_per_tick
    manipulated_jobs = sum(1 for job in selected if job.manipulation is not None)
    minimal_tool_calls = facts.horizon_ticks + (3 * len(selected)) + manipulated_jobs
    expected_tool_burn = Decimal(minimal_tool_calls) * facts.tool_call_cost
    return (gross_profit - expected_overhead - expected_tool_burn).quantize(Decimal("0.01"))
