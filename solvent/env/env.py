from __future__ import annotations

import hashlib
from dataclasses import asdict
from decimal import Decimal
from typing import Any

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
from solvent.trace.writer import TraceWriter

ARTIFACT_PREVIEW_LIMIT = 4096


class Environment:
    """Coordinates Solvent episodes and the public tool boundary."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.clock = Clock(config.horizon_ticks)
        self.ledger = Ledger(config.start_balance)
        self.market = Market(
            seed=config.seed,
            version=config.market_version,
            market_size=config.market_size,
            decoy_rate=config.decoy_rate,
            redteam_enabled=config.redteam_enabled,
        )
        self.trace = TraceWriter(config.trace_path)
        self.bid_jobs: set[str] = set()
        self.accepted_jobs: dict[str, AcceptedJob] = {}
        self.pending_manipulations: dict[str, PendingManipulation] = {}
        self.submitted_jobs: set[str] = set()
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
                "overhead_per_tick": config.overhead_per_tick,
                "tool_call_cost": config.tool_call_cost,
                "market_version": config.market_version,
                "market_size": config.market_size,
                "decoy_rate": config.decoy_rate,
                "redteam_enabled": config.redteam_enabled,
            },
            Decimal("0"),
        )
        self._evaluate_termination()

    def list_jobs(self) -> list[JobPublic]:
        self._charge_tool_or_raise("list_jobs")
        jobs = [
            job.to_public()
            for job in self.market.available_jobs(self.clock.tick)
            if job.id not in self.bid_jobs
        ]
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
        if job_id in self.bid_jobs:
            self._invalid(
                "bid",
                f"job already has a final bid: {job_id}",
                self.config.tool_call_cost,
                code="duplicate_bid",
                payload={"job_id": job_id},
            )

        self.bid_jobs.add(job_id)
        self._emit("bid_made", {"job_id": job_id, "price": price}, self.config.tool_call_cost)
        accepted = price <= job.reservation_price
        public_manipulation = None
        if accepted:
            self.accepted_jobs[job_id] = AcceptedJob(job_id=job_id, contract_price=price)
            self._emit("bid_accepted", {"job_id": job_id, "contract_price": price}, Decimal("0"))
            if job.manipulation is not None:
                self.pending_manipulations[job_id] = PendingManipulation(job_id=job_id, event=job.manipulation)
                public_manipulation = job.manipulation.to_public()
                self._emit("manipulation_attempt", {"job_id": job_id, **public_manipulation}, Decimal("0"))
        else:
            self._emit("bid_declined", {"job_id": job_id, "price": price}, Decimal("0"))
        self._evaluate_termination()
        return {"accepted": accepted, "job_id": job_id, "manipulation": public_manipulation}

    def submit(self, job_id: str, artifact: str) -> VerifyResult:
        self._charge_tool_or_raise("submit", {"job_id": job_id})
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

    def end_tick(self) -> None:
        if self.terminated_reason is not None:
            raise AlreadyTerminatedError("episode already terminated")
        self.ledger.debit_overhead(self.config.overhead_per_tick)
        self._emit("overhead_charged", {"amount": self.config.overhead_per_tick}, self.config.overhead_per_tick)
        new_tick = self.clock.advance()
        self._emit("tick_advanced", {"tick": new_tick}, Decimal("0"))
        self._evaluate_termination()

    def terminated(self) -> bool:
        return self.terminated_reason is not None

    def finalize(self) -> EpisodeSummary:
        if self._summary is not None:
            return self._summary
        if self.terminated_reason is None:
            self.terminated_reason = "turn_cap" if self.clock.reached_horizon() else "turn_cap"
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
        if job.arrival_tick > self.clock.tick:
            raise UnknownJobError(f"job has not arrived yet: {job_id}")
        return job

    def _verify(self, job: Job, artifact: str) -> VerifyResult:
        if job.type != "data_clean":
            raise InvalidActionError(f"unsupported job type: {job.type}")
        return verify_data_clean(job, artifact)

    def _evaluate_termination(self) -> None:
        if self.terminated_reason is not None:
            return
        if self.ledger.insolvent():
            self.terminated_reason = "insolvent"
        elif self.clock.reached_horizon():
            self.terminated_reason = "turn_cap"

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
