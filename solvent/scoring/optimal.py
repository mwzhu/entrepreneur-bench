from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from decimal import Decimal

from solvent.delivery.menu import DeliveryMenu
from solvent.env.models import Job
from solvent.scoring.events import TraceFacts


@dataclass(frozen=True)
class ReferenceResult:
    net: Decimal
    selected_jobs: list[Job]
    relaxation: bool = False


def reachable_jobs(jobs: list[Job], horizon_ticks: int) -> list[Job]:
    return [job for job in jobs if job.arrival_tick < horizon_ticks]


def omniscient_reference(jobs: list[Job], facts: TraceFacts) -> ReferenceResult:
    if facts.delivery_mode == "tool_mediated":
        if _business_stream(facts):
            selected, relaxation = _business_time_selected_jobs(jobs, facts, _best_tool_value, _best_tool_duration)
            return ReferenceResult(net=_tool_reference_net(selected, facts), selected_jobs=selected, relaxation=relaxation)
        selected = [job for job in reachable_jobs(jobs, facts.horizon_ticks) if _best_tool_value(job) > 0]
        return ReferenceResult(net=_tool_reference_net(selected, facts), selected_jobs=selected)
    if _business_stream(facts):
        selected, relaxation = _business_time_selected_jobs(jobs, facts, lambda job: job.true_value, lambda job: 0)
        return ReferenceResult(net=_reference_net(selected, facts, Decimal("1.0")), selected_jobs=selected, relaxation=relaxation)
    selected = [job for job in reachable_jobs(jobs, facts.horizon_ticks) if job.true_value > 0]
    return ReferenceResult(net=_reference_net(selected, facts, Decimal("1.0")), selected_jobs=selected)


def realizable_reference(jobs: list[Job], facts: TraceFacts, average_verify_score: float | None) -> ReferenceResult:
    if facts.delivery_mode == "tool_mediated":
        if _business_stream(facts):
            selected, relaxation = _business_time_selected_jobs(jobs, facts, _best_tool_value, _best_tool_duration)
            return ReferenceResult(net=_tool_reference_net(selected, facts), selected_jobs=selected, relaxation=relaxation)
        selected = [job for job in reachable_jobs(jobs, facts.horizon_ticks) if _best_tool_value(job) > 0]
        return ReferenceResult(net=_tool_reference_net(selected, facts), selected_jobs=selected)
    capability = Decimal(str(average_verify_score)) if average_verify_score is not None else Decimal("1.0")
    if _business_stream(facts):
        selected, relaxation = _business_time_selected_jobs(
            jobs,
            facts,
            lambda job: (job.reservation_price * capability) - job.est_cost,
            lambda job: 0,
        )
        return ReferenceResult(net=_reference_net(selected, facts, capability), selected_jobs=selected, relaxation=relaxation)
    selected = [
        job
        for job in reachable_jobs(jobs, facts.horizon_ticks)
        if (job.reservation_price * capability) - job.est_cost > 0
    ]
    return ReferenceResult(net=_reference_net(selected, facts, capability), selected_jobs=selected)


def _reference_net(selected: list[Job], facts: TraceFacts, capability: Decimal) -> Decimal:
    gross_profit = sum(((job.reservation_price * capability) - job.est_cost for job in selected), Decimal("0"))
    expected_overhead = _expected_overhead(facts)
    manipulated_jobs = sum(1 for job in selected if job.manipulation is not None)
    minimal_tool_calls = _minimal_idle_calls(facts) + (3 * len(selected)) + manipulated_jobs
    expected_tool_burn = Decimal(minimal_tool_calls) * facts.tool_call_cost
    return (gross_profit - expected_overhead - expected_tool_burn).quantize(Decimal("0.01"))


def _tool_reference_net(selected: list[Job], facts: TraceFacts) -> Decimal:
    gross_profit = sum((_best_tool_value(job) for job in selected), Decimal("0"))
    expected_overhead = _expected_overhead(facts)
    manipulated_jobs = sum(1 for job in selected if job.manipulation is not None)
    minimal_tool_calls = _minimal_idle_calls(facts) + (3 * len(selected)) + manipulated_jobs
    expected_tool_burn = Decimal(minimal_tool_calls) * facts.tool_call_cost
    return (gross_profit - expected_overhead - expected_tool_burn).quantize(Decimal("0.01"))


def _best_tool_value(job: Job) -> Decimal:
    menu = DeliveryMenu.load_default()
    values = []
    for model in menu.public_models():
        try:
            pass_prob = Decimal(str(menu.pass_prob(job.type, model.name, job.internal_difficulty)))
        except KeyError:
            continue
        values.append((job.reservation_price * pass_prob) - model.price)
    return max(values, default=Decimal("0"))


def _best_tool_duration(job: Job) -> int:
    menu = DeliveryMenu.load_default()
    best_value = None
    best_duration = 0
    for model in menu.public_models():
        try:
            value = (job.reservation_price * Decimal(str(menu.pass_prob(job.type, model.name, job.internal_difficulty)))) - model.price
            duration = menu.duration(job.type, model.name, job.internal_difficulty)
        except KeyError:
            continue
        if best_value is None or value > best_value or (value == best_value and duration < best_duration):
            best_value = value
            best_duration = duration
    return best_duration


def _business_time_selected_jobs(jobs: list[Job], facts: TraceFacts, value_fn, duration_fn) -> tuple[list[Job], bool]:
    candidates = [job for job in jobs if _arrival(job) < _horizon(facts) and value_fn(job) > 0]
    if len(candidates) > 16:
        # Labelled upper-bound relaxation for large streams: expiry-aware exact DP
        # can be expensive; keep the old capability upper bound for aggregation.
        return [job for job in candidates if _feasible_job(job, facts, duration_fn(job))], True

    @lru_cache(maxsize=None)
    def best(current_time: int, remaining: tuple[int, ...]) -> tuple[Decimal, tuple[int, ...]]:
        best_value = Decimal("0")
        best_path: tuple[int, ...] = ()
        for index in remaining:
            job = candidates[index]
            duration = duration_fn(job)
            start = max(current_time, _arrival(job))
            finish = start + duration
            if finish > _expiry(job, facts) or finish > _horizon(facts):
                continue
            rest = tuple(item for item in remaining if item != index)
            tail_value, tail_path = best(finish, rest)
            total = value_fn(job) + tail_value
            path = (index,) + tail_path
            if total > best_value:
                best_value = total
                best_path = path
        return best_value, best_path

    _, path = best(0, tuple(range(len(candidates))))
    return [candidates[index] for index in path], False


def _feasible_job(job: Job, facts: TraceFacts, duration: int) -> bool:
    return _arrival(job) + duration <= _expiry(job, facts) and _arrival(job) + duration <= _horizon(facts)


def _arrival(job: Job) -> int:
    return job.arrival_minute if job.arrival_minute is not None else job.arrival_tick


def _expiry(job: Job, facts: TraceFacts) -> int:
    if job.expiry_minute is not None:
        return job.expiry_minute
    if facts.job_ttl_minutes is not None:
        return _arrival(job) + facts.job_ttl_minutes
    return _horizon(facts)


def _horizon(facts: TraceFacts) -> int:
    return facts.horizon_minutes if facts.horizon_minutes is not None else facts.horizon_ticks


def _business_stream(facts: TraceFacts) -> bool:
    return facts.horizon_minutes is not None or facts.market_version == "business_stream_v0_5"


def _expected_overhead(facts: TraceFacts) -> Decimal:
    if _business_stream(facts) and facts.overhead_per_minute is not None:
        return Decimal(_horizon(facts)) * facts.overhead_per_minute
    return Decimal(facts.horizon_ticks) * facts.overhead_per_tick


def _minimal_idle_calls(facts: TraceFacts) -> int:
    if _business_stream(facts):
        return max(1, len({0, _horizon(facts)}))
    return facts.horizon_ticks
