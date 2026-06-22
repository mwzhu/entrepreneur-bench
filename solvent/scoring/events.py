from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass
class BidFact:
    job_id: str
    price: Decimal
    accepted: bool | None = None
    # Negotiation terminal outcome for the counter mechanic (v0.6). `outcome` is one
    # of: counter_accepted, counter_rejected, counter_rejected_then_floor,
    # counter_rejected_then_declined. starting_price/counter_price/contract_price are
    # filled from the acceptance / rejection events.
    starting_price: Decimal | None = None
    counter_price: Decimal | None = None
    contract_price: Decimal | None = None
    outcome: str | None = None


@dataclass(frozen=True)
class AcceptedJobFact:
    """A job the agent actually took, built from terminal acceptance events.

    Covers both floor accepts (`job_accepted`, via_counter=False) and winning
    counters (`bid_accepted`, via_counter=True). `contract_price` is the resolved
    revenue the job pays on a successful delivery, available even when delivery
    later fails (no `paid` event is emitted in that case).
    """

    job_id: str
    contract_price: Decimal
    starting_price: Decimal
    via_counter: bool


@dataclass(frozen=True)
class SubmissionFact:
    job_id: str
    passed: bool
    score: float


@dataclass(frozen=True)
class DeliveryAttemptFact:
    job_id: str
    mode: str
    model: str | None
    passed: bool
    score: float
    price_charged: Decimal
    duration: int
    attempt_index: int = 0


@dataclass(frozen=True)
class TraceFacts:
    seed: int
    config_id: str
    start_balance: Decimal
    horizon_ticks: int
    horizon_minutes: int | None
    overhead_per_tick: Decimal
    overhead_per_minute: Decimal | None
    tool_call_cost: Decimal
    market_version: str
    market_size: int
    arrival_rate_per_day: Decimal
    job_ttl_minutes: int | None
    decoy_rate: Decimal
    redteam_enabled: bool
    end_balance: Decimal
    net_revenue: Decimal
    tool_burn: Decimal
    overhead_burn: Decimal
    bids: list[BidFact]
    submissions: list[SubmissionFact]
    accepted_jobs: dict[str, "AcceptedJobFact"]
    declined_jobs: set[str]
    counter_rejected_jobs: set[str]
    paid_revenue_by_job: dict[str, Decimal]
    manipulation_attempts: set[str]
    manipulation_conceded: dict[str, Decimal]
    manipulation_resisted: set[str]
    invalid_actions: list[dict[str, Any]]
    terminated_reason: str
    compatibility_estimated_horizon: bool
    brain_tokens_in: int = 0
    brain_tokens_out: int = 0
    brain_cache_read_tokens: int = 0
    brain_cache_write_tokens: int = 0
    brain_cost: Decimal = Decimal("0")
    delivery_mode: str = "direct"
    menu_version: str = "menu_v0_4"
    menu_checksum: str = ""
    task_mix: dict[str, float] | None = None
    difficulty_distribution: dict[str, float] | None = None
    seed_split: str = "ad_hoc"
    pricing_table_version: str = "pricing_v0_4"
    corpus_schema_version: str = "none"
    menu_schema_version: str = "solvent_delivery_menu_v0_4"
    delivery_attempts: list[DeliveryAttemptFact] | None = None
    # Tick the episode actually reached (max event tick). Equals horizon_ticks for a
    # full run; lower when the agent stopped early (turn/budget cap, insolvency). Used
    # to compute truncation-free "realized horizon" reference optima.
    realized_horizon_ticks: int = 0


def load_events(trace_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def facts_from_events(events: list[dict[str, Any]]) -> TraceFacts:
    if not events:
        raise ValueError("trace contains no events")
    start = events[0]
    start_payload = start.get("payload", {})
    seed = int(start_payload["seed"])
    config_id = str(start_payload["config_id"])
    start_balance = _decimal(start_payload["start_balance"])
    metadata_v0_2 = "market_version" in start_payload
    market_version = str(start_payload.get("market_version", "data_clean_static_v0_1"))
    market_size = int(start_payload.get("market_size", 3))
    decoy_rate = _decimal(start_payload.get("decoy_rate", "0"))
    redteam_enabled = bool(start_payload.get("redteam_enabled", False))
    horizon_ticks = int(start_payload.get("horizon_ticks", events[-1]["tick"]))
    horizon_minutes = _optional_int(start_payload.get("horizon_minutes"))
    realized_horizon_ticks = min(horizon_ticks, max((int(event.get("tick", 0)) for event in events), default=horizon_ticks))
    compatibility_estimated_horizon = not metadata_v0_2
    overhead_per_tick = _infer_overhead(events, start_payload)
    overhead_per_minute = _optional_decimal(start_payload.get("overhead_per_minute"))
    tool_call_cost = _infer_tool_call_cost(events, start_payload)
    arrival_rate_per_day = _decimal(start_payload.get("arrival_rate_per_day", "1.00"))
    provenance = start_payload.get("provenance", {})
    delivery_mode = str(provenance.get("delivery_mode", "direct"))
    menu_version = str(provenance.get("menu_version", "menu_v0_4"))
    menu_checksum = str(provenance.get("menu_checksum", ""))
    task_mix = _float_dict(provenance.get("task_mix", {"data_clean": 1.0}))
    difficulty_distribution = _float_dict(provenance.get("difficulty_distribution", {"easy": 1.0}))
    seed_split = str(provenance.get("seed_split", "ad_hoc"))
    pricing_table_version = str(provenance.get("pricing_table_version", "pricing_v0_4"))
    corpus_schema_version = str(provenance.get("corpus_schema_version", "none"))
    menu_schema_version = str(provenance.get("menu_schema_version", "solvent_delivery_menu_v0_4"))
    job_ttl_minutes = _optional_int(provenance.get("job_ttl_minutes"))

    bids: list[BidFact] = []
    bids_by_job: dict[str, BidFact] = {}
    submissions: list[SubmissionFact] = []
    delivery_attempts: list[DeliveryAttemptFact] = []
    pending_deliveries: dict[str, dict[str, Any]] = {}
    accepted_jobs: dict[str, AcceptedJobFact] = {}
    declined_jobs: set[str] = set()
    counter_rejected_jobs: set[str] = set()
    paid_revenue_by_job: dict[str, Decimal] = {}
    manipulation_attempts: set[str] = set()
    manipulation_conceded: dict[str, Decimal] = {}
    manipulation_resisted: set[str] = set()
    invalid_actions: list[dict[str, Any]] = []
    tool_burn = Decimal("0")
    overhead_burn = Decimal("0")
    terminated_reason = ""
    brain_tokens_in = 0
    brain_tokens_out = 0
    brain_cache_read_tokens = 0
    brain_cache_write_tokens = 0
    brain_cost = Decimal("0")

    for event in events:
        kind = event["kind"]
        payload = event.get("payload", {})
        burn = _decimal(event.get("burn_delta", "0"))
        if kind == "overhead_charged":
            overhead_burn += burn
        elif burn > 0:
            tool_burn += burn

        if kind == "bid_made":
            bid = BidFact(job_id=str(payload["job_id"]), price=_decimal(payload["price"]))
            bids.append(bid)
            bids_by_job[bid.job_id] = bid
        elif kind == "bid_accepted":
            job_id = str(payload["job_id"])
            contract_price = _decimal(payload.get("contract_price", payload.get("counter_price", "0")))
            starting_price = _optional_decimal(payload.get("starting_price"))
            if starting_price is None:
                starting_price = contract_price
            accepted_jobs[job_id] = AcceptedJobFact(
                job_id=job_id,
                contract_price=contract_price,
                starting_price=starting_price,
                via_counter=bool(payload.get("counter_accepted", True)),
            )
            counter_rejected_jobs.discard(job_id)
            declined_jobs.discard(job_id)
            bid = bids_by_job.get(job_id)
            if bid is not None:
                bid.accepted = True
                bid.contract_price = contract_price
                bid.counter_price = _optional_decimal(payload.get("counter_price"))
                bid.starting_price = starting_price
                bid.outcome = "counter_accepted"
        elif kind == "job_accepted":
            job_id = str(payload["job_id"])
            contract_price = _decimal(payload.get("contract_price", payload.get("starting_price", "0")))
            starting_price = _optional_decimal(payload.get("starting_price"))
            if starting_price is None:
                starting_price = contract_price
            accepted_jobs[job_id] = AcceptedJobFact(
                job_id=job_id,
                contract_price=contract_price,
                starting_price=starting_price,
                via_counter=False,
            )
            had_rejected_counter = job_id in counter_rejected_jobs
            counter_rejected_jobs.discard(job_id)
            declined_jobs.discard(job_id)
            bid = bids_by_job.get(job_id)
            if bid is not None:
                bid.accepted = True
                bid.contract_price = contract_price
                bid.starting_price = starting_price
                bid.outcome = "counter_rejected_then_floor" if had_rejected_counter else "floor_accepted"
        elif kind == "counter_rejected":
            job_id = str(payload["job_id"])
            counter_rejected_jobs.add(job_id)
            bid = bids_by_job.get(job_id)
            if bid is not None:
                bid.accepted = False
                bid.counter_price = _optional_decimal(payload.get("counter_price"))
                bid.starting_price = _optional_decimal(payload.get("starting_price"))
                bid.outcome = "counter_rejected"
        elif kind == "job_declined":
            job_id = str(payload["job_id"])
            declined_jobs.add(job_id)
            if job_id in counter_rejected_jobs:
                counter_rejected_jobs.discard(job_id)
                bid = bids_by_job.get(job_id)
                if bid is not None:
                    bid.outcome = "counter_rejected_then_declined"
        elif kind == "bid_declined":
            # Legacy (pre-v0.6) one-shot decline; treat as a rejected attempt.
            if payload["job_id"] in bids_by_job:
                bids_by_job[payload["job_id"]].accepted = False
        elif kind in {"verified_pass", "verified_fail"}:
            submission = SubmissionFact(
                job_id=str(payload["job_id"]),
                passed=kind == "verified_pass",
                score=float(payload.get("score", 1.0 if kind == "verified_pass" else 0.0)),
            )
            submissions.append(submission)
            delivery_attempts.append(
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
            )
        elif kind == "delivered":
            pending_deliveries[str(payload["job_id"])] = payload
        elif kind in {"delivery_passed", "delivery_failed"}:
            job_id = str(payload["job_id"])
            delivered = pending_deliveries.get(job_id, {})
            delivery_attempts.append(
                DeliveryAttemptFact(
                    job_id=job_id,
                    mode="tool_mediated",
                    model=str(payload.get("model", delivered.get("model", ""))),
                    passed=kind == "delivery_passed",
                    score=float(payload.get("score", 1.0 if kind == "delivery_passed" else 0.0)),
                    price_charged=_decimal(payload.get("price_charged", delivered.get("price_charged", "0"))),
                    duration=int(payload.get("duration", delivered.get("duration", 0))),
                    attempt_index=int(payload.get("attempt_index", delivered.get("attempt_index", 0))),
                )
            )
        elif kind == "paid":
            paid_revenue_by_job[str(payload["job_id"])] = _decimal(payload["revenue"])
        elif kind == "manipulation_attempt":
            manipulation_attempts.add(str(payload["job_id"]))
        elif kind == "manipulation_conceded":
            manipulation_conceded[str(payload["job_id"])] = _decimal(payload["conceded_value"])
        elif kind == "manipulation_resisted":
            manipulation_resisted.add(str(payload["job_id"]))
        elif kind == "invalid_action":
            invalid_actions.append(payload)
        elif kind == "brain_metered":
            brain_tokens_in = int(payload.get("cumulative_input_tokens", brain_tokens_in))
            brain_tokens_out = int(payload.get("cumulative_output_tokens", brain_tokens_out))
            brain_cache_read_tokens = int(payload.get("cumulative_cache_read_tokens", brain_cache_read_tokens))
            brain_cache_write_tokens = int(payload.get("cumulative_cache_write_tokens", brain_cache_write_tokens))
            brain_cost = _decimal(payload.get("cumulative_cost", brain_cost))
        elif kind == "terminated":
            terminated_reason = str(payload.get("reason", ""))

    end_balance = _decimal(events[-1]["balance_after"])
    return TraceFacts(
        seed=seed,
        config_id=config_id,
        start_balance=start_balance,
        horizon_ticks=horizon_ticks,
        horizon_minutes=horizon_minutes,
        realized_horizon_ticks=realized_horizon_ticks,
        overhead_per_tick=overhead_per_tick,
        overhead_per_minute=overhead_per_minute,
        tool_call_cost=tool_call_cost,
        market_version=market_version,
        market_size=market_size,
        arrival_rate_per_day=arrival_rate_per_day,
        job_ttl_minutes=job_ttl_minutes,
        decoy_rate=decoy_rate,
        redteam_enabled=redteam_enabled,
        end_balance=end_balance,
        net_revenue=end_balance - start_balance,
        tool_burn=tool_burn,
        overhead_burn=overhead_burn,
        bids=bids,
        submissions=submissions,
        accepted_jobs=accepted_jobs,
        declined_jobs=declined_jobs,
        counter_rejected_jobs=counter_rejected_jobs,
        paid_revenue_by_job=paid_revenue_by_job,
        manipulation_attempts=manipulation_attempts,
        manipulation_conceded=manipulation_conceded,
        manipulation_resisted=manipulation_resisted,
        invalid_actions=invalid_actions,
        terminated_reason=terminated_reason,
        compatibility_estimated_horizon=compatibility_estimated_horizon,
        brain_tokens_in=brain_tokens_in,
        brain_tokens_out=brain_tokens_out,
        brain_cache_read_tokens=brain_cache_read_tokens,
        brain_cache_write_tokens=brain_cache_write_tokens,
        brain_cost=brain_cost,
        delivery_mode=delivery_mode,
        menu_version=menu_version,
        menu_checksum=menu_checksum,
        task_mix=task_mix,
        difficulty_distribution=difficulty_distribution,
        seed_split=seed_split,
        pricing_table_version=pricing_table_version,
        corpus_schema_version=corpus_schema_version,
        menu_schema_version=menu_schema_version,
        delivery_attempts=delivery_attempts,
    )


def _infer_overhead(events: list[dict[str, Any]], start_payload: dict[str, Any]) -> Decimal:
    if "overhead_per_tick" in start_payload:
        return _decimal(start_payload["overhead_per_tick"])
    for event in events:
        if event["kind"] == "overhead_charged":
            return _decimal(event.get("payload", {}).get("amount", event.get("burn_delta", "0.05")))
    return Decimal("0.05")


def _infer_tool_call_cost(events: list[dict[str, Any]], start_payload: dict[str, Any]) -> Decimal:
    if "tool_call_cost" in start_payload:
        return _decimal(start_payload["tool_call_cost"])
    for event in events:
        if event["kind"] != "overhead_charged":
            burn = _decimal(event.get("burn_delta", "0"))
            if burn > 0:
                return burn
    return Decimal("0.01")


def _decimal(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _optional_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(val) for key, val in value.items()}
