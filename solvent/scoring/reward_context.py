from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent.env.models import Job
from solvent.scoring.events import AcceptedJobFact, TraceFacts


@dataclass(frozen=True)
class RewardContext:
    trace_path: Path
    facts: TraceFacts
    jobs_by_id: dict[str, Job]
    accepted_facts: dict[str, AcceptedJobFact]
    delivered_job_ids: set[str]
    good_ids: set[str]
    delivery_menu: Any
    expected_net_revenue: Decimal
    oracle_tool_regret: Decimal
    delivered_selection_regret: Decimal
    terminated_reason: str


def pricing_regret_over(
    job_ids: set[str],
    accepted_facts: dict[str, AcceptedJobFact],
    jobs_by_id: dict[str, Job],
    good_ids: set[str],
) -> Decimal:
    surplus_left = Decimal("0")
    for job_id in job_ids:
        fact = accepted_facts.get(job_id)
        job = jobs_by_id.get(job_id)
        if fact is None or job is None or job_id not in good_ids:
            continue
        surplus_left += job.reservation_price - fact.contract_price
    return surplus_left.quantize(Decimal("0.01"))


def selection_regret_over(
    chosen_ids: set[str],
    good_ids: set[str],
    jobs_by_id: dict[str, Job],
    reachable_by_id: dict[str, Job],
    optimal_value: Decimal,
    scheduled_selection_value: Callable[[list[Job]], Decimal],
    job_selection_value: Callable[[Job], Decimal],
) -> Decimal:
    good_chosen = [job_id for job_id in chosen_ids if job_id in good_ids and job_id in jobs_by_id]
    decoys_chosen = [
        job_id
        for job_id in chosen_ids
        if job_id in reachable_by_id and job_id not in good_ids
    ]
    chosen_value = scheduled_selection_value([jobs_by_id[job_id] for job_id in good_chosen])
    missed_good = max(Decimal("0"), optimal_value - chosen_value)
    chased_decoys = sum(
        (abs(job_selection_value(reachable_by_id[job_id])) for job_id in decoys_chosen),
        Decimal("0"),
    )
    return (missed_good + chased_decoys).quantize(Decimal("0.01"))
