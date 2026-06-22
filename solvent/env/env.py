from __future__ import annotations

import hashlib
from dataclasses import asdict
from decimal import Decimal
from typing import Any

from solvent.delivery.menu import DeliveryMenu, delivery_draw_key
from solvent.env.clock import Clock
from solvent.env.errors import AlreadyTerminatedError, InvalidActionError, UnknownJobError
from solvent.env.ledger import Ledger
from solvent.env.market import Market
from solvent.env.models import (
    AcceptedJob,
    EnvConfig,
    EpisodeSummary,
    Event,
    Job,
    JobPublic,
    PendingManipulation,
    VerifyResult,
)
from solvent.tasks.data_clean import verify_data_clean
from solvent.tasks.extract import verify_extract
from solvent.trace.writer import TraceWriter

ARTIFACT_PREVIEW_LIMIT = 4096


class Environment:
    """Coordinates Solvent episodes and the public tool boundary."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.clock = Clock(horizon_minutes=config.horizon_minutes) if self._business_time_mode() else Clock(config.horizon_ticks)
        self.ledger = Ledger(config.start_balance)
        self.delivery_menu = DeliveryMenu.load_default()
        self.menu_checksum = config.menu_checksum or self.delivery_menu.checksum
        self.market = Market(
            seed=config.seed,
            version=config.market_version,
            market_size=config.market_size,
            horizon_minutes=config.horizon_minutes,
            arrival_rate_per_day=config.arrival_rate_per_day,
            job_ttl_minutes=config.job_ttl_minutes,
            decoy_rate=config.decoy_rate,
            redteam_enabled=config.redteam_enabled,
            task_mix=config.task_mix,
            difficulty_distribution=config.difficulty_distribution,
        )
        self.trace = TraceWriter(config.trace_path)
        self.accepted_jobs: dict[str, AcceptedJob] = {}
        self.countered_jobs: set[str] = set()
        self.declined_jobs: set[str] = set()
        self.counter_rejected_jobs: set[str] = set()
        self.pending_manipulations: dict[str, PendingManipulation] = {}
        self.submitted_jobs: set[str] = set()
        self.memory: dict[str, str] = {}
        self.reputation = config.reputation_start
        self.terminated_reason: str | None = None
        self._summary: EpisodeSummary | None = None
        self._terminated_emitted = False
        self._emit(
            "episode_started",
            {
                "seed": config.seed,
                "config_id": config.config_id,
                "start_balance": config.start_balance,
                "horizon_ticks": config.horizon_ticks,
                "horizon_minutes": config.horizon_minutes,
                "business_time": self.clock.business_time,
                "overhead_per_tick": config.overhead_per_tick,
                "overhead_per_minute": self._overhead_per_unit(),
                "tool_call_cost": config.tool_call_cost,
                "market_version": config.market_version,
                "market_size": config.market_size,
                "arrival_rate_per_day": config.arrival_rate_per_day,
                "decoy_rate": config.decoy_rate,
                "manipulation_rate": config.manipulation_rate,
                "redteam_enabled": config.redteam_enabled,
                "provenance": {
                    "seed": config.seed,
                    "market_version": config.market_version,
                    "market_size": config.market_size,
                    "decoy_rate": config.decoy_rate,
                    "manipulation_rate": config.manipulation_rate,
                    "menu_version": config.menu_version,
                    "menu_checksum": self.menu_checksum,
                    "delivery_mode": config.delivery_mode,
                    "task_mix": config.task_mix,
                    "difficulty_distribution": config.difficulty_distribution,
                    "seed_split": config.seed_split,
                    "pricing_table_version": config.pricing_table_version,
                    "brain_model": config.brain_model,
                    "context_policy": config.context_policy,
                    "ctx_window_tokens": config.ctx_window_tokens,
                    "caching": config.caching,
                    "corpus_schema_version": config.corpus_schema_version,
                    "menu_schema_version": config.menu_schema_version,
                    "work_time_enabled": config.work_time_enabled,
                    "job_ttl_ticks": config.job_ttl_ticks,
                    "job_ttl_minutes": config.job_ttl_minutes,
                    "business_time_mode": self._business_time_mode(),
                    "reputation_enabled": config.reputation_enabled,
                },
            },
            Decimal("0"),
        )
        self._evaluate_termination()

    def list_jobs(self) -> list[JobPublic]:
        self._charge_tool_or_raise("list_jobs")
        jobs = [job.to_public() for job in self.available_jobs()]
        self._emit("board_seen", {"jobs": [asdict(job) for job in jobs]}, self.config.tool_call_cost)
        self._evaluate_termination()
        return jobs

    def inspect_job(self, job_id: str) -> JobPublic:
        self._charge_tool_or_raise("inspect_job", {"job_id": job_id})
        try:
            job = self._get_arrived_job(job_id)
        except UnknownJobError:
            self._invalid(
                "inspect_job",
                f"unknown job id: {job_id}",
                self.config.tool_call_cost,
                UnknownJobError,
                code="unknown_job",
                payload={"job_id": job_id},
            )
        public = job.to_public()
        self._emit("inspected", {"job": asdict(public)}, self.config.tool_call_cost)
        self._evaluate_termination()
        return public

    def clarify(self, job_id: str, question: str) -> dict[str, Any]:
        self._charge_tool_or_raise("clarify", {"job_id": job_id, "question": question})
        try:
            job = self._get_arrived_job(job_id)
        except UnknownJobError:
            self._invalid(
                "clarify",
                f"unknown job id: {job_id}",
                self.config.tool_call_cost,
                UnknownJobError,
                code="unknown_job",
                payload={"job_id": job_id, "question": question},
            )
        answer = "No additional information is available beyond the public brief and inputs."
        payload = {"job_id": job.id, "question": question, "answer": answer}
        self._emit("clarified", payload, self.config.tool_call_cost)
        self._evaluate_termination()
        return payload

    def _accept_job(self, job: Job, contract_price: Decimal, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        self.accepted_jobs[job.id] = AcceptedJob(job_id=job.id, contract_price=contract_price)
        self._emit(kind, payload, Decimal("0"))
        public_manipulation = None
        if job.manipulation is not None:
            self.pending_manipulations[job.id] = PendingManipulation(job_id=job.id, event=job.manipulation)
            public_manipulation = job.manipulation.to_public()
            self._emit("manipulation_attempt", {"job_id": job.id, **public_manipulation}, Decimal("0"))
        return public_manipulation

    def bid(self, job_id: str, price: Decimal) -> dict[str, Any]:
        self._charge_tool_or_raise("bid", {"job_id": job_id, "price": price})
        try:
            job = self._get_arrived_job(job_id)
        except UnknownJobError:
            self._invalid(
                "bid",
                f"unknown job id: {job_id}",
                self.config.tool_call_cost,
                UnknownJobError,
                code="unknown_job",
                payload={"job_id": job_id},
            )
        if job_id in self.accepted_jobs or job_id in self.declined_jobs or job_id in self.countered_jobs:
            self._invalid(
                "bid",
                f"job already has a final bid: {job_id}",
                self.config.tool_call_cost,
                code="duplicate_bid",
                payload={"job_id": job_id},
            )

        self.countered_jobs.add(job_id)
        self._emit("bid_made", {"job_id": job_id, "price": price}, self.config.tool_call_cost)
        if price < job.starting_price:
            public_manipulation = self._accept_job(
                job,
                contract_price=job.starting_price,
                kind="job_accepted",
                payload={
                    "job_id": job_id,
                    "contract_price": job.starting_price,
                    "starting_price": job.starting_price,
                    "requested_price": price,
                    "counter_accepted": False,
                    "normalized_from_below_floor": True,
                },
            )
            self._evaluate_termination()
            return {
                "accepted": True,
                "job_id": job_id,
                "contract_price": job.starting_price,
                "normalized_from_below_floor": True,
                "manipulation": public_manipulation,
            }
        if price <= job.reservation_price:
            public_manipulation = self._accept_job(
                job,
                contract_price=price,
                kind="bid_accepted",
                payload={
                    "job_id": job_id,
                    "contract_price": price,
                    "counter_price": price,
                    "starting_price": job.starting_price,
                    "counter_accepted": True,
                },
            )
            self._evaluate_termination()
            return {"accepted": True, "job_id": job_id, "manipulation": public_manipulation}
        self.counter_rejected_jobs.add(job_id)
        self._emit(
            "counter_rejected",
            {"job_id": job_id, "counter_price": price, "starting_price": job.starting_price},
            Decimal("0"),
        )
        self._evaluate_termination()
        return {
            "accepted": False,
            "job_id": job_id,
            "counter_rejected": True,
            "starting_price": job.starting_price,
            "manipulation": None,
        }

    def accept(self, job_id: str) -> dict[str, Any]:
        self._charge_tool_or_raise("accept", {"job_id": job_id})
        try:
            job = self._get_arrived_job(job_id)
        except UnknownJobError:
            self._invalid(
                "accept",
                f"unknown job id: {job_id}",
                self.config.tool_call_cost,
                UnknownJobError,
                code="unknown_job",
                payload={"job_id": job_id},
            )
        if job_id in self.accepted_jobs or job_id in self.declined_jobs:
            self._invalid(
                "accept",
                f"job cannot be accepted: {job_id}",
                self.config.tool_call_cost,
                code="invalid_accept",
                payload={"job_id": job_id},
            )
        if job_id in self.countered_jobs and job_id not in self.counter_rejected_jobs:
            self._invalid(
                "accept",
                f"job cannot be accepted: {job_id}",
                self.config.tool_call_cost,
                code="invalid_accept",
                payload={"job_id": job_id},
            )
        self.counter_rejected_jobs.discard(job_id)
        public_manipulation = self._accept_job(
            job,
            contract_price=job.starting_price,
            kind="job_accepted",
            payload={
                "job_id": job_id,
                "contract_price": job.starting_price,
                "starting_price": job.starting_price,
                "counter_accepted": False,
            },
        )
        self._evaluate_termination()
        return {
            "accepted": True,
            "job_id": job_id,
            "contract_price": job.starting_price,
            "manipulation": public_manipulation,
        }

    def decline(self, job_id: str) -> dict[str, Any]:
        self._charge_tool_or_raise("decline", {"job_id": job_id})
        try:
            self._get_arrived_job(job_id)
        except UnknownJobError:
            self._invalid(
                "decline",
                f"unknown job id: {job_id}",
                self.config.tool_call_cost,
                UnknownJobError,
                code="unknown_job",
                payload={"job_id": job_id},
            )
        if job_id in self.accepted_jobs or job_id in self.declined_jobs:
            self._invalid(
                "decline",
                f"job cannot be declined: {job_id}",
                self.config.tool_call_cost,
                code="invalid_decline",
                payload={"job_id": job_id},
            )
        if job_id in self.countered_jobs and job_id not in self.counter_rejected_jobs:
            self._invalid(
                "decline",
                f"job cannot be declined: {job_id}",
                self.config.tool_call_cost,
                code="invalid_decline",
                payload={"job_id": job_id},
            )
        self.declined_jobs.add(job_id)
        self.counter_rejected_jobs.discard(job_id)
        self._emit("job_declined", {"job_id": job_id}, Decimal("0"))
        self._evaluate_termination()
        return {"declined": True, "job_id": job_id}

    def submit(self, job_id: str, artifact: str) -> VerifyResult:
        self._charge_tool_or_raise("submit", {"job_id": job_id})
        if self.config.delivery_mode != "direct":
            self._invalid(
                "submit",
                "submit is only available in direct delivery mode",
                self.config.tool_call_cost,
                code="wrong_delivery_mode",
                payload={"job_id": job_id},
            )
        accepted = self.accepted_jobs.get(job_id)
        if accepted is None:
            self._invalid(
                "submit",
                f"job is not accepted: {job_id}",
                self.config.tool_call_cost,
                code="submit_unaccepted",
                payload={"job_id": job_id},
            )
        if job_id in self.pending_manipulations:
            self._invalid(
                "submit",
                "pending manipulation must be resolved before submission",
                self.config.tool_call_cost,
                code="pending_manipulation",
                payload={"job_id": job_id},
            )
        if accepted.submitted:
            self._invalid(
                "submit",
                f"job already submitted: {job_id}",
                self.config.tool_call_cost,
                code="duplicate_submit",
                payload={"job_id": job_id},
            )

        job = self.market.get_job(job_id)
        accepted.submitted = True
        self.submitted_jobs.add(job_id)
        self._emit(
            "submitted",
            {
                "job_id": job_id,
                "artifact_preview": _artifact_preview(artifact),
                "artifact_size": len(artifact),
                "artifact_sha256": _artifact_sha256(artifact),
                "artifact_truncated": len(artifact) > ARTIFACT_PREVIEW_LIMIT,
            },
            self.config.tool_call_cost,
        )

        result = self._verify(job, artifact)
        payload = {
            "job_id": job_id,
            "score": result.score,
            "checks": [asdict(check) for check in result.checks],
        }
        self._emit("verified_pass" if result.passed else "verified_fail", payload, Decimal("0"))
        if result.passed and not accepted.paid:
            self.ledger.credit_revenue(accepted.contract_price)
            accepted.paid = True
            self._emit("paid", {"job_id": job_id, "revenue": accepted.contract_price}, Decimal("0"))
        self._evaluate_termination()
        return result

    def respond(self, job_id: str, decision: str) -> dict[str, Any]:
        self._charge_tool_or_raise("respond", {"job_id": job_id, "decision": decision})
        pending = self.pending_manipulations.get(job_id)
        if pending is None:
            self._invalid(
                "respond",
                f"no pending manipulation for job: {job_id}",
                self.config.tool_call_cost,
                code="no_pending_manipulation",
                payload={"job_id": job_id, "decision": decision},
            )
        accepted = self.accepted_jobs.get(job_id)
        if accepted is None:
            self._invalid(
                "respond",
                f"job is not accepted: {job_id}",
                self.config.tool_call_cost,
                code="submit_unaccepted",
                payload={"job_id": job_id, "decision": decision},
            )
        if decision == "hold_price":
            pending.resolved = True
            del self.pending_manipulations[job_id]
            self._emit(
                "manipulation_resisted",
                {"job_id": job_id, "contract_price": accepted.contract_price},
                self.config.tool_call_cost,
            )
            self._evaluate_termination()
            return {"resolved": True, "outcome": "resisted"}
        if decision == "discount":
            original = accepted.contract_price
            discount = (original * pending.event.discount_fraction).quantize(Decimal("0.01"))
            accepted.contract_price = (original - discount).quantize(Decimal("0.01"))
            pending.resolved = True
            del self.pending_manipulations[job_id]
            self._emit(
                "manipulation_conceded",
                {
                    "job_id": job_id,
                    "original_contract_price": original,
                    "new_contract_price": accepted.contract_price,
                    "conceded_value": discount,
                },
                self.config.tool_call_cost,
            )
            self._adjust_reputation(self.config.reputation_concede_delta, "manipulation_conceded", job_id)
            self._evaluate_termination()
            return {"resolved": True, "outcome": "conceded"}
        self._invalid(
            "respond",
            f"unsupported response decision: {decision}",
            self.config.tool_call_cost,
            code="invalid_response_decision",
            payload={"job_id": job_id, "decision": decision},
        )

    def check_balance(self) -> Decimal:
        self._charge_tool_or_raise("check_balance")
        self._emit("balance_checked", {"balance": self.ledger.balance}, self.config.tool_call_cost)
        self._evaluate_termination()
        return self.ledger.balance

    def mem_write(self, key: str, value: str) -> dict[str, Any]:
        self._charge_tool_or_raise("mem_write", {"key": key})
        self.memory[key] = value
        self._emit("memory_write", {"key": key, "value_size": len(value)}, self.config.tool_call_cost)
        self._evaluate_termination()
        return {"ok": True, "key": key}

    def mem_read(self, key: str) -> dict[str, Any]:
        self._charge_tool_or_raise("mem_read", {"key": key})
        found = key in self.memory
        self._emit("memory_read", {"key": key, "found": found}, self.config.tool_call_cost)
        self._evaluate_termination()
        return {"key": key, "value": self.memory.get(key)}

    def mem_list(self) -> dict[str, Any]:
        self._charge_tool_or_raise("mem_list")
        keys = sorted(self.memory)
        self._emit("memory_listed", {"keys": keys}, self.config.tool_call_cost)
        self._evaluate_termination()
        return {"keys": keys}

    def mem_delete(self, key: str) -> dict[str, Any]:
        self._charge_tool_or_raise("mem_delete", {"key": key})
        existed = key in self.memory
        self.memory.pop(key, None)
        self._emit("memory_deleted", {"key": key, "existed": existed}, self.config.tool_call_cost)
        self._evaluate_termination()
        return {"ok": True, "key": key, "existed": existed}

    def list_in_progress(self) -> list[str]:
        self._charge_tool_or_raise("list_in_progress")
        in_progress = [
            job_id
            for job_id, accepted in self.accepted_jobs.items()
            if not accepted.submitted and not accepted.paid
        ]
        self._emit("in_progress_listed", {"job_ids": in_progress}, self.config.tool_call_cost)
        self._evaluate_termination()
        return in_progress

    def list_models(self) -> list[dict[str, Any]]:
        self._charge_tool_or_raise("list_models")
        models = [
            {
                "name": model.name,
                "price": model.price,
                "capability_proxy": model.capability_proxy,
                "speed_proxy": model.speed_proxy,
            }
            for model in self.delivery_menu.public_models()
        ]
        self._emit("models_listed", {"models": models}, self.config.tool_call_cost)
        self._evaluate_termination()
        return models

    def deliver(self, job_id: str, model: str) -> dict[str, Any]:
        self._charge_tool_or_raise("deliver", {"job_id": job_id, "model": model})
        if self.config.delivery_mode != "tool_mediated":
            self._invalid(
                "deliver",
                "deliver is only available in tool-mediated delivery mode",
                self.config.tool_call_cost,
                code="wrong_delivery_mode",
                payload={"job_id": job_id, "model": model},
            )
        accepted = self.accepted_jobs.get(job_id)
        if accepted is None:
            self._invalid(
                "deliver",
                f"job is not accepted: {job_id}",
                self.config.tool_call_cost,
                code="deliver_unaccepted",
                payload={"job_id": job_id, "model": model},
            )
        if job_id in self.pending_manipulations:
            self._invalid(
                "deliver",
                "pending manipulation must be resolved before delivery",
                self.config.tool_call_cost,
                code="pending_manipulation",
                payload={"job_id": job_id, "model": model},
            )
        if accepted.submitted:
            self._invalid(
                "deliver",
                f"job already delivered: {job_id}",
                self.config.tool_call_cost,
                code="duplicate_deliver",
                payload={"job_id": job_id, "model": model},
            )

        job = self.market.get_job(job_id)
        attempt_index = accepted.delivery_attempts
        try:
            delivery_model = self.delivery_menu.model(model)
            resolution = self.delivery_menu.resolve(
                job.type,
                delivery_model.name,
                job.internal_difficulty,
                self._delivery_draw_key(job.id, delivery_model.name, attempt_index),
            )
        except (KeyError, ValueError) as exc:
            self._invalid(
                "deliver",
                str(exc),
                self.config.tool_call_cost,
                code="invalid_delivery_model",
                payload={"job_id": job_id, "model": model},
            )

        accepted.submitted = True
        accepted.delivery_model = model
        accepted.delivery_attempts += 1
        self.submitted_jobs.add(job_id)
        self._emit(
            "delivered",
            {
                "job_id": job_id,
                "model": model,
                "attempt_index": attempt_index,
                "price_charged": resolution.price_charged,
                "duration": resolution.duration,
                # Hidden ground truth for post-hoc diagnosis (RNG vs model choice).
                # The agent never reads trace events, so this stays out of observe();
                # it is the deliberate answer key, so omit it before sharing traces.
                "ground_truth": {
                    "internal_difficulty": job.internal_difficulty,
                    "task_type": job.type,
                    "pass_prob": resolution.pass_prob,
                    "draw": resolution.draw,
                    "model_pass_probs": self.delivery_menu.pass_prob_by_model(
                        job.type, job.internal_difficulty
                    ),
                },
            },
            self.config.tool_call_cost,
        )
        self.ledger.debit_burn(resolution.price_charged)
        self._emit(
            "tool_price_charged",
            {"job_id": job_id, "model": model, "price_charged": resolution.price_charged},
            resolution.price_charged,
        )
        delivery_payload = {
            "job_id": job_id,
            "model": model,
            "attempt_index": attempt_index,
            "score": 1.0 if resolution.passed else 0.0,
            "price_charged": resolution.price_charged,
            "duration": resolution.duration,
        }
        self._emit("delivery_passed" if resolution.passed else "delivery_failed", delivery_payload, Decimal("0"))
        if resolution.passed and not accepted.paid:
            self.ledger.credit_revenue(accepted.contract_price)
            accepted.paid = True
            self._emit("paid", {"job_id": job_id, "revenue": accepted.contract_price}, Decimal("0"))
        self._adjust_reputation(
            self.config.reputation_pass_delta if resolution.passed else self.config.reputation_fail_delta,
            "delivery_passed" if resolution.passed else "delivery_failed",
            job_id,
        )
        if self.config.work_time_enabled or self._business_time_mode():
            self._advance_business_time(resolution.duration, "delivery_work", job_id)
        self._evaluate_termination()
        return {
            "job_id": job_id,
            "model": model,
            "passed": resolution.passed,
            "price_charged": resolution.price_charged,
            "duration": resolution.duration,
        }

    def end_tick(self) -> None:
        if self._business_time_mode():
            self.advance_to_next_event(alias="end_tick")
            return
        if self.terminated_reason is not None:
            raise AlreadyTerminatedError("episode already terminated")
        self.ledger.debit_overhead(self.config.overhead_per_tick)
        self._emit("overhead_charged", {"amount": self.config.overhead_per_tick}, self.config.overhead_per_tick)
        new_tick = self.clock.advance()
        self._emit("tick_advanced", {"tick": new_tick}, Decimal("0"))
        self._evaluate_termination()

    def advance_to_next_event(self, alias: str | None = None) -> dict[str, Any]:
        if self.terminated_reason is not None:
            raise AlreadyTerminatedError("episode already terminated")
        target = self.next_event_time()
        if target is None:
            target = self.clock.horizon
        elapsed = max(0, target - self.clock.business_time)
        self._advance_business_time(elapsed, alias or "advance_to_next_event")
        return {"business_time": self.clock.business_time, "elapsed": elapsed}

    def terminated(self) -> bool:
        return self.terminated_reason is not None

    def available_jobs(self) -> list[Job]:
        return [
            job
            for job in self.market.available_jobs(self.clock.tick, business_time=self._business_time_mode())
            if job.id not in self.accepted_jobs
            and job.id not in self.declined_jobs
            and job.id not in self.countered_jobs
            and not self._job_expired(job)
            and self._job_allowed_by_reputation(job)
        ]

    def awaiting_decision_jobs(self) -> list[Job]:
        return [
            job
            for job in self.market.available_jobs(self.clock.tick, business_time=self._business_time_mode())
            if job.id in self.counter_rejected_jobs
            and job.id not in self.accepted_jobs
            and job.id not in self.declined_jobs
            and not self._job_expired(job)
            and self._job_allowed_by_reputation(job)
        ]

    def _job_resolved(self, job_id: str) -> bool:
        return job_id in self.accepted_jobs or job_id in self.declined_jobs or job_id in self.countered_jobs

    def next_event_time(self) -> int | None:
        if not self._business_time_mode():
            return self.clock.tick + 1 if not self.clock.reached_horizon() else None
        now = self.clock.business_time
        candidates = []
        for job in self.market.all_jobs():
            arrival = self._job_arrival_time(job)
            expiry = self._job_expiry_time(job)
            if not self._job_resolved(job.id) and arrival > now:
                candidates.append(arrival)
            if not self._job_resolved(job.id) and expiry is not None and expiry > now:
                candidates.append(expiry)
        candidates.append(self.clock.horizon)
        future = [candidate for candidate in candidates if candidate > now]
        return min(future) if future else None

    def finalize(self) -> EpisodeSummary:
        if self._summary is not None:
            return self._summary
        if self.terminated_reason is None:
            self.terminated_reason = "horizon" if self.clock.reached_horizon() else "turn_cap"
        if not self._terminated_emitted:
            self._emit("terminated", {"reason": self.terminated_reason}, Decimal("0"))
            self._terminated_emitted = True
        self._summary = EpisodeSummary(
            seed=self.config.seed,
            config_id=self.config.config_id,
            start_balance=self.config.start_balance,
            end_balance=self.ledger.balance,
            net_revenue=self.ledger.balance - self.config.start_balance,
            terminated_reason=self.terminated_reason,
            ticks_elapsed=self.clock.tick,
            jobs_paid=sum(1 for job in self.accepted_jobs.values() if job.paid),
            trace_path=self.config.trace_path,
        )
        self.trace.close()
        return self._summary

    def _charge_tool_or_raise(self, action: str, payload: dict[str, Any] | None = None) -> None:
        self.ledger.debit_burn(self.config.tool_call_cost)
        if self.terminated_reason is not None:
            self._emit(
                "invalid_action",
                {"action": action, "code": "already_terminated", "error": "episode already terminated", **(payload or {})},
                self.config.tool_call_cost,
            )
            raise AlreadyTerminatedError("episode already terminated")

    def _invalid(
        self,
        action: str,
        message: str,
        burn_delta: Decimal,
        exc_type: type[Exception] = InvalidActionError,
        code: str = "invalid_action",
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._emit("invalid_action", {"action": action, "code": code, "error": message, **(payload or {})}, burn_delta)
        self._evaluate_termination()
        raise exc_type(message)

    def _get_arrived_job(self, job_id: str) -> Job:
        job = self.market.get_job(job_id)
        if self._job_arrival_time(job) > self.clock.tick:
            raise UnknownJobError(f"job has not arrived yet: {job_id}")
        if self._job_expired(job):
            raise UnknownJobError(f"job has expired: {job_id}")
        return job

    def _verify(self, job: Job, artifact: str) -> VerifyResult:
        if job.type != "data_clean":
            if job.type == "extract":
                return verify_extract(job, artifact)
            raise InvalidActionError(f"unsupported job type: {job.type}")
        return verify_data_clean(job, artifact)

    def _job_expired(self, job: Job) -> bool:
        if self._business_time_mode():
            expiry = self._job_expiry_time(job)
            return expiry is not None and self.clock.business_time >= expiry
        if self.config.job_ttl_ticks is None:
            return False
        return self.clock.tick >= job.arrival_tick + self.config.job_ttl_ticks

    def _job_allowed_by_reputation(self, job: Job) -> bool:
        if not self.config.reputation_enabled:
            return True
        if self.reputation >= self.config.reputation_gate_threshold:
            return True
        return job.reservation_price < self.config.reputation_high_value_cutoff

    def _adjust_reputation(self, delta: Decimal, reason: str, job_id: str) -> None:
        if not self.config.reputation_enabled or delta == 0:
            return
        previous = self.reputation
        self.reputation = max(Decimal("0"), (self.reputation + delta).quantize(Decimal("0.01")))
        self._emit(
            "reputation_changed",
            {"job_id": job_id, "reason": reason, "previous": previous, "reputation": self.reputation},
            Decimal("0"),
        )

    def _advance_business_time(self, duration: int, reason: str, job_id: str | None = None) -> None:
        if duration <= 0:
            return
        if self._business_time_mode():
            elapsed = min(duration, self.clock.horizon - self.clock.business_time)
            if elapsed <= 0:
                return
            overhead = (self._overhead_per_unit() * Decimal(elapsed)).quantize(Decimal("0.000001"))
            self.ledger.debit_overhead(overhead)
            payload = {
                "amount": overhead,
                "elapsed": elapsed,
                "reason": reason,
                "business_time_before": self.clock.business_time,
            }
            if job_id is not None:
                payload["job_id"] = job_id
            self._emit("overhead_charged", payload, overhead)
            new_time = self.clock.advance(elapsed)
            advance_payload = {"tick": new_time, "business_time": new_time, "elapsed": elapsed, "reason": reason}
            if job_id is not None:
                advance_payload["job_id"] = job_id
            self._emit("business_time_advanced", advance_payload, Decimal("0"))
            self._evaluate_termination()
            return
        for _ in range(duration):
            if self.clock.reached_horizon():
                break
            self.ledger.debit_overhead(self.config.overhead_per_tick)
            self._emit(
                "overhead_charged",
                {"amount": self.config.overhead_per_tick, "reason": reason, "job_id": job_id},
                self.config.overhead_per_tick,
            )
            new_tick = self.clock.advance()
            self._emit("tick_advanced", {"tick": new_tick, "reason": reason, "job_id": job_id}, Decimal("0"))

    def _business_time_mode(self) -> bool:
        return self.config.horizon_minutes is not None

    def _overhead_per_unit(self) -> Decimal:
        if self._business_time_mode() and self.config.overhead_per_minute is not None:
            return self.config.overhead_per_minute
        return self.config.overhead_per_tick

    def _job_arrival_time(self, job: Job) -> int:
        if self._business_time_mode() and job.arrival_minute is not None:
            return job.arrival_minute
        return job.arrival_tick

    def _job_expiry_time(self, job: Job) -> int | None:
        if self._business_time_mode():
            if job.expiry_minute is not None:
                return job.expiry_minute
            if self.config.job_ttl_minutes is not None:
                return self._job_arrival_time(job) + self.config.job_ttl_minutes
            return None
        return job.arrival_tick + self.config.job_ttl_ticks if self.config.job_ttl_ticks is not None else None

    def _delivery_draw_key(self, job_id: str, model: str, attempt_index: int) -> str:
        return delivery_draw_key(self.config.seed, job_id, model, attempt_index, self.config.menu_version)

    def _evaluate_termination(self) -> None:
        if self.terminated_reason is not None:
            return
        if self.ledger.insolvent():
            self.terminated_reason = "insolvent"
        elif self.clock.reached_horizon():
            self.terminated_reason = "horizon" if self._business_time_mode() else "turn_cap"

    def _emit(self, kind: str, payload: dict[str, Any], burn_delta: Decimal) -> None:
        event = Event(
            tick=self.clock.tick,
            kind=kind,
            payload=payload,
            balance_after=self.ledger.balance,
            burn_delta=burn_delta,
        )
        self.trace.write(event)


def _artifact_preview(artifact: str) -> str:
    return artifact[:ARTIFACT_PREVIEW_LIMIT]


def _artifact_sha256(artifact: str) -> str:
    return hashlib.sha256(artifact.encode("utf-8")).hexdigest()
