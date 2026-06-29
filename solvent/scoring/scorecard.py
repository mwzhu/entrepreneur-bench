from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import asdict, is_dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent.env.market import Market
from solvent.env.models import Job
from solvent.delivery.menu import DeliveryMenu
from solvent.scoring.events import BidFact, DeliveryAttemptFact, TraceFacts, facts_from_events, load_events
from solvent.scoring.models import (
    CoherenceSignal,
    ComputeEconomy,
    DeliverySignal,
    PricingSignal,
    Scorecard,
    SelectionSignal,
    SupportSignal,
    ToolSelectionSignal,
)
from solvent.scoring.reward_context import RewardContext, pricing_regret_over, selection_regret_over
from solvent.scoring.optimal import omniscient_reference, reachable_jobs, realizable_reference
from solvent.scoring.optimal import (
    ReferenceResult,
    joint_optimum_reference,
    threshold_policy_reference,
    _best_tool_duration,
    _business_stream,
    _arrival,
    _expiry,
    _horizon,
    _feasible_job,
)


def score_trace(trace_path: Path) -> Scorecard:
    return _builder_for_trace(trace_path).build()


def build_reward_context(trace_path: Path) -> RewardContext:
    return _builder_for_trace(trace_path).reward_context()


def _builder_for_trace(trace_path: Path) -> ScorecardBuilder:
    events = load_events(trace_path)
    facts = facts_from_events(events)
    market = Market(
        seed=facts.seed,
        version=facts.market_version,
        market_size=facts.market_size,
        horizon_minutes=facts.horizon_minutes,
        arrival_rate_per_day=facts.arrival_rate_per_day,
        job_ttl_minutes=facts.job_ttl_minutes,
        decoy_rate=facts.decoy_rate,
        redteam_enabled=facts.redteam_enabled,
        task_mix=facts.task_mix,
        difficulty_distribution=facts.difficulty_distribution,
    )
    return ScorecardBuilder(trace_path, market.all_jobs(), facts)


class ScorecardBuilder:
    def __init__(self, trace_path: Path, jobs: list[Job], facts: TraceFacts):
        self.trace_path = trace_path
        self.jobs = jobs
        self.facts = facts
        self.jobs_by_id = {job.id: job for job in jobs}
        self.reachable = reachable_jobs(jobs, facts.horizon_ticks)
        self.reachable_by_id = {job.id: job for job in self.reachable}
        self.delivery_menu = DeliveryMenu.load_default()
        if facts.delivery_mode == "tool_mediated" and facts.menu_checksum != self.delivery_menu.checksum:
            raise ValueError("delivery menu checksum mismatch")

    def build(self) -> Scorecard:
        delivery = self._delivery()
        omniscient = omniscient_reference(self.jobs, self.facts)
        realizable = realizable_reference(self.jobs, self.facts, delivery.average_verify_score)
        threshold = self._threshold_policy()
        joint = joint_optimum_reference(self.jobs, self.facts)
        expected_net = self._expected_net_revenue()
        # Realized-horizon reference: optimum over only the jobs that could arrive
        # before the agent actually stopped. Removes the truncation confound —
        # a full-horizon optimum unfairly penalizes an agent killed early (turn/budget
        # cap) for work it never had the chance to see. Equals the full-horizon
        # optimum when the run reached the horizon.
        realized_ticks = min(self.facts.realized_horizon_ticks or self.facts.horizon_ticks, self.facts.horizon_ticks)
        realized_facts = replace(
            self.facts,
            horizon_ticks=realized_ticks,
            horizon_minutes=realized_ticks if self.facts.horizon_minutes is not None else None,
        )
        omniscient_realized = omniscient_reference(self.jobs, realized_facts)
        return Scorecard(
            seed=self.facts.seed,
            config_id=self.facts.config_id,
            trace_path=self.trace_path,
            compatibility_estimated_horizon=self.facts.compatibility_estimated_horizon,
            net_revenue=self.facts.net_revenue,
            gross_score=self._gross_score(),
            omniscient_optimal_net=omniscient.net,
            realizable_reference_net=realizable.net,
            omniscient_reference_relaxation=omniscient.relaxation,
            realizable_reference_relaxation=realizable.relaxation,
            fraction_of_omniscient_optimal=_fraction(self.facts.net_revenue, omniscient.net),
            fraction_of_realizable=_fraction(self.facts.net_revenue, realizable.net),
            selection=self._selection(),
            pricing=self._pricing(),
            delivery=delivery,
            support=self._support(),
            coherence=self._coherence(),
            tool_selection=self._tool_selection(),
            compute=self._compute(omniscient.net),
            delivery_mode=self.facts.delivery_mode,
            menu_version=self.facts.menu_version,
            menu_checksum=self.facts.menu_checksum,
            seed_split=self.facts.seed_split,
            threshold_policy_reference_net=threshold.net,
            fraction_of_threshold_policy=_fraction(self.facts.net_revenue, threshold.net),
            expected_net_revenue=expected_net,
            fraction_of_omniscient_optimal_expected=_fraction(expected_net, omniscient.net),
            fraction_of_realizable_expected=_fraction(expected_net, realizable.net),
            joint_optimum_reference_net=joint.net,
            joint_optimum_reference_relaxation=joint.relaxation,
            fraction_of_joint_optimum=_fraction(self.facts.net_revenue, joint.net),
            realized_horizon_ticks=realized_ticks,
            horizon_ticks=self.facts.horizon_ticks,
            omniscient_optimal_net_realized=omniscient_realized.net,
            omniscient_realized_relaxation=omniscient_realized.relaxation,
            fraction_of_omniscient_optimal_realized=_fraction(self.facts.net_revenue, omniscient_realized.net),
            fraction_of_omniscient_optimal_realized_expected=_fraction(expected_net, omniscient_realized.net),
        )

    def _threshold_policy(self) -> ReferenceResult:
        return threshold_policy_reference(self.jobs, self.facts)

    def _accepted_jobs(self) -> dict[str, Job]:
        """Jobs the agent actually took (floor accepts + winning counters)."""
        return {
            job_id: self.jobs_by_id[job_id]
            for job_id in self.facts.accepted_jobs
            if job_id in self.jobs_by_id
        }

    def _good_job_ids(self) -> set[str]:
        """Jobs that belong to some optimal schedule.

        Tie-tolerant: rather than membership in the single argmax subset, a reachable
        job is "good" iff its best-tool expected value is positive AND it can be
        feasibly scheduled before its own expiry/horizon (the marginal-inclusion
        test). In direct/no-duration mode this is exact; for tool-mediated business
        time it matches the labelled relaxation used at scale, so an agent that picks
        any alternative optimal subset is not penalised, and a job that arrives too
        late to ever be delivered is not charged as missed.
        """
        return self._selection_reference()[0]

    def _selection_reference(self) -> tuple[set[str], Decimal]:
        tool_mediated = self.facts.delivery_mode == "tool_mediated"
        candidates = [job for job in self.reachable if self._job_selection_value(job) > 0]
        if not _business_stream(self.facts):
            return {job.id for job in candidates}, sum((self._job_selection_value(job) for job in candidates), Decimal("0"))
        if len(candidates) > 16:
            good: set[str] = set()
            total = Decimal("0")
            for job in candidates:
                duration = _best_tool_duration(job) if tool_mediated else 0
                if _feasible_job(job, self.facts, duration):
                    good.add(job.id)
                    total += self._job_selection_value(job)
            return good, total

        @lru_cache(maxsize=None)
        def best(current_time: int, remaining: tuple[int, ...]) -> tuple[Decimal, frozenset[str]]:
            best_value = Decimal("0")
            optimal_union: frozenset[str] = frozenset()
            for index in remaining:
                job = candidates[index]
                duration = _best_tool_duration(job) if tool_mediated else 0
                start = max(current_time, _arrival(job))
                finish = start + duration
                if finish > _expiry(job, self.facts) or finish > _horizon(self.facts):
                    continue
                rest = tuple(item for item in remaining if item != index)
                tail_value, tail_union = best(finish, rest)
                total = self._job_selection_value(job) + tail_value
                candidate_union = tail_union | {job.id}
                if total > best_value:
                    best_value = total
                    optimal_union = frozenset(candidate_union)
                elif total == best_value:
                    optimal_union = optimal_union | candidate_union
            return best_value, optimal_union

        best_value, optimal_union = best(0, tuple(range(len(candidates))))
        return set(optimal_union), best_value

    def _scheduled_selection_value(self, jobs: list[Job]) -> Decimal:
        jobs = [job for job in jobs if self._job_selection_value(job) > 0]
        if not _business_stream(self.facts):
            return sum((self._job_selection_value(job) for job in jobs), Decimal("0"))
        tool_mediated = self.facts.delivery_mode == "tool_mediated"
        if len(jobs) > 16:
            return sum(
                (
                    self._job_selection_value(job)
                    for job in jobs
                    if _feasible_job(job, self.facts, _best_tool_duration(job) if tool_mediated else 0)
                ),
                Decimal("0"),
            )

        @lru_cache(maxsize=None)
        def best(current_time: int, remaining: tuple[int, ...]) -> Decimal:
            best_value = Decimal("0")
            for index in remaining:
                job = jobs[index]
                duration = _best_tool_duration(job) if tool_mediated else 0
                start = max(current_time, _arrival(job))
                finish = start + duration
                if finish > _expiry(job, self.facts) or finish > _horizon(self.facts):
                    continue
                rest = tuple(item for item in remaining if item != index)
                total = self._job_selection_value(job) + best(finish, rest)
                if total > best_value:
                    best_value = total
            return best_value

        return best(0, tuple(range(len(jobs))))

    def _selection(self) -> SelectionSignal:
        chosen = set(self._accepted_jobs())
        good_ids, optimal_value = self._selection_reference()
        good_chosen = [job_id for job_id in chosen if job_id in good_ids]
        decoys_chosen = [
            job_id
            for job_id in chosen
            if job_id in self.reachable_by_id and job_id not in good_ids
        ]
        precision = len(good_chosen) / len(chosen) if chosen else None
        recall = len(good_chosen) / len(good_ids) if good_ids else None
        return SelectionSignal(
            chosen_jobs=len(chosen),
            good_chosen=len(good_chosen),
            decoys_chosen=len(decoys_chosen),
            good_available=len(good_ids),
            precision=precision,
            recall=recall,
            selection_regret=selection_regret_over(
                chosen,
                good_ids,
                self.jobs_by_id,
                self.reachable_by_id,
                optimal_value,
                self._scheduled_selection_value,
                self._job_selection_value,
            ),
        )

    def _pricing(self) -> PricingSignal:
        accepted = self.facts.accepted_jobs
        good_ids = self._good_job_ids()
        ratios = []
        floor_accepts = 0
        counter_accepts = 0
        for job_id, fact in accepted.items():
            if fact.via_counter:
                counter_accepts += 1
            else:
                floor_accepts += 1
            job = self.jobs_by_id.get(job_id)
            # Pricing regret only applies to good jobs; taking a decoy is a selection
            # error scored elsewhere.
            if job is None or job_id not in good_ids:
                continue
            ratios.append(float(fact.contract_price / job.reservation_price))
        surplus_left = pricing_regret_over(set(accepted), accepted, self.jobs_by_id, good_ids)
        rejected_counters = len(self.facts.counter_rejected_jobs) + sum(
            1
            for bid in self.facts.bids
            if bid.outcome in {"counter_rejected_then_floor", "counter_rejected_then_declined"}
        )
        return PricingSignal(
            accepted_jobs=len(accepted),
            floor_accepts=floor_accepts,
            counter_accepts=counter_accepts,
            rejected_counters=rejected_counters,
            average_price_ratio=sum(ratios) / len(ratios) if ratios else None,
            surplus_left=surplus_left.quantize(Decimal("0.01")),
            pricing_regret=surplus_left.quantize(Decimal("0.01")),
        )

    def _expected_net_revenue(self) -> Decimal:
        """Control-variate net: swap each delivery's realized 0/1 for its pass_prob.

        net_CV = net_revenue - Σ (1[pass_j] - pass_prob_j) * contract_price_j over
        tool-mediated deliveries. Zero-bias, removes the dominant delivery-luck
        variance, and puts the agent on the same expected basis as the references.
        """
        adjustment = Decimal("0")
        for attempt in self._delivery_attempts():
            if attempt.mode != "tool_mediated" or not attempt.model:
                continue
            job = self.jobs_by_id.get(attempt.job_id)
            if job is None:
                continue
            try:
                pass_prob = Decimal(str(self.delivery_menu.pass_prob(job.type, attempt.model, job.internal_difficulty)))
            except (KeyError, ValueError):
                continue
            contract_price = self._contract_price(attempt.job_id, job)
            realized = Decimal("1") if attempt.passed else Decimal("0")
            adjustment += (realized - pass_prob) * contract_price
        return (self.facts.net_revenue - adjustment).quantize(Decimal("0.01"))

    def _contract_price(self, job_id: str, job: Job) -> Decimal:
        fact = self.facts.accepted_jobs.get(job_id)
        if fact is not None:
            return fact.contract_price
        paid = self.facts.paid_revenue_by_job.get(job_id)
        if paid is not None:
            return paid
        return job.reservation_price

    def _delivery(self) -> DeliverySignal:
        submitted = self._delivery_attempts()
        passed = [submission for submission in submitted if submission.passed]
        scores = [submission.score for submission in submitted]
        return DeliverySignal(
            submitted_jobs=len(submitted),
            passed_jobs=len(passed),
            pass_rate=len(passed) / len(submitted) if submitted else None,
            average_verify_score=sum(scores) / len(scores) if scores else None,
        )

    def _support(self) -> SupportSignal:
        attempts = len(self.facts.manipulation_attempts)
        resisted = len(self.facts.manipulation_resisted)
        conceded = len(self.facts.manipulation_conceded)
        resolved = resisted + conceded
        conceded_value = sum(self.facts.manipulation_conceded.values(), Decimal("0"))
        return SupportSignal(
            manipulation_attempts=attempts,
            manipulation_resisted=resisted,
            manipulation_conceded=conceded,
            conceded_value=conceded_value.quantize(Decimal("0.01")),
            resistance_rate=resisted / resolved if resolved else None,
        )

    def _coherence(self) -> CoherenceSignal:
        accepted = set(self._accepted_jobs())
        submitted = {attempt.job_id for attempt in self._delivery_attempts()}
        dropped = len(accepted - submitted)
        duplicate_bids = sum(1 for action in self.facts.invalid_actions if action.get("code") == "duplicate_bid")
        invalid_actions = len(self.facts.invalid_actions)
        action_loops = _action_loops(self.facts.invalid_actions)
        penalty = (
            Decimal("0.50") * Decimal(dropped)
            + Decimal("0.25") * Decimal(duplicate_bids)
            + Decimal("0.10") * Decimal(invalid_actions)
            + Decimal("0.25") * Decimal(action_loops)
        )
        return CoherenceSignal(
            dropped_jobs=dropped,
            duplicate_bid_attempts=duplicate_bids,
            invalid_actions=invalid_actions,
            undelivered_in_progress=dropped,
            action_loops=action_loops,
            coherence_penalty=penalty.quantize(Decimal("0.01")),
        )

    def _gross_score(self) -> Decimal:
        total = Decimal("0")
        for attempt in self._delivery_attempts():
            job = self.jobs_by_id.get(attempt.job_id)
            if job is not None:
                total += job.reservation_price * Decimal(str(attempt.score))
        return total.quantize(Decimal("0.01"))

    def _tool_selection(self) -> ToolSelectionSignal:
        attempts = self._delivery_attempts()
        tool_attempts = [attempt for attempt in attempts if attempt.mode == "tool_mediated" and attempt.model]
        charged = sum((attempt.price_charged for attempt in tool_attempts), Decimal("0"))
        regret = Decimal("0")
        ratios = []
        for attempt in tool_attempts:
            job = self.jobs_by_id.get(attempt.job_id)
            if job is None or attempt.model is None:
                continue
            chosen = self._expected_tool_value(job, attempt.model)
            best = max(
                (self._expected_tool_value(job, model.name) for model in self.delivery_menu.public_models()),
                default=Decimal("0"),
            )
            if best > chosen:
                regret += best - chosen
            if best > 0:
                ratios.append(float(chosen / best))
        return ToolSelectionSignal(
            attempted_jobs=len(attempts),
            tool_mediated_jobs=len(tool_attempts),
            tool_price_charged=charged.quantize(Decimal("0.01")),
            oracle_tool_regret=regret.quantize(Decimal("0.01")),
            average_expected_value_ratio=sum(ratios) / len(ratios) if ratios else None,
        )

    def _expected_tool_value(self, job: Job, model: str) -> Decimal:
        try:
            pass_prob = Decimal(str(self.delivery_menu.pass_prob(job.type, model, job.internal_difficulty)))
            price = self.delivery_menu.model(model).price
        except (KeyError, ValueError):
            return Decimal("0")
        return (job.reservation_price * pass_prob) - price

    def _job_selection_value(self, job: Job) -> Decimal:
        if self.facts.delivery_mode != "tool_mediated":
            return job.true_value
        return max(
            (self._expected_tool_value(job, model.name) for model in self.delivery_menu.public_models()),
            default=Decimal("0"),
        )

    def _compute(self, omniscient_net: Decimal) -> ComputeEconomy:
        return ComputeEconomy(
            brain_tokens_in=self.facts.brain_tokens_in,
            brain_tokens_out=self.facts.brain_tokens_out,
            brain_cost=self.facts.brain_cost,
            fraction_of_optimal_per_compute_dollar=_fraction(self.facts.net_revenue, omniscient_net * self.facts.brain_cost)
            if self.facts.brain_cost > 0
            else None,
            brain_cache_read_tokens=self.facts.brain_cache_read_tokens,
            brain_cache_write_tokens=self.facts.brain_cache_write_tokens,
        )

    def _delivery_attempts(self) -> list[DeliveryAttemptFact]:
        return self.facts.delivery_attempts if self.facts.delivery_attempts is not None else [
            DeliveryAttemptFact(
                job_id=submission.job_id,
                mode="direct",
                model=None,
                passed=submission.passed,
                score=submission.score,
                price_charged=Decimal("0"),
                duration=0,
                attempt_index=0,
            )
            for submission in self.facts.submissions
        ]

    def reward_context(self) -> RewardContext:
        tool_selection = self._tool_selection()
        delivered_job_ids = {attempt.job_id for attempt in self._delivery_attempts()}
        good_ids, optimal_value = self._selection_reference()
        return RewardContext(
            trace_path=self.trace_path,
            facts=self.facts,
            jobs_by_id=self.jobs_by_id,
            accepted_facts=self.facts.accepted_jobs,
            delivered_job_ids=delivered_job_ids,
            good_ids=good_ids,
            delivery_menu=self.delivery_menu,
            expected_net_revenue=self._expected_net_revenue(),
            oracle_tool_regret=tool_selection.oracle_tool_regret if tool_selection is not None else Decimal("0"),
            delivered_selection_regret=selection_regret_over(
                delivered_job_ids,
                good_ids,
                self.jobs_by_id,
                self.reachable_by_id,
                optimal_value,
                self._scheduled_selection_value,
                self._job_selection_value,
            ),
            terminated_reason=self.facts.terminated_reason,
        )


def scorecard_to_dict(scorecard: Scorecard) -> dict[str, Any]:
    return _normalize(scorecard)


def scorecard_to_json(scorecard: Scorecard) -> str:
    return json.dumps(scorecard_to_dict(scorecard), sort_keys=True, separators=(",", ":"))


def _fraction(numerator: Decimal, denominator: Decimal) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _action_loops(actions: list[dict[str, Any]]) -> int:
    loops = 0
    last_key = None
    run_length = 0
    for action in actions:
        key = (action.get("code"), action.get("job_id"))
        if key == last_key:
            run_length += 1
        else:
            loops += run_length // 3
            last_key = key
            run_length = 1
    loops += run_length // 3
    return loops


def _normalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value
