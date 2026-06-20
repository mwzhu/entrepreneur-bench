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


@dataclass(frozen=True)
class SubmissionFact:
    job_id: str
    passed: bool
    score: float


@dataclass(frozen=True)
class TraceFacts:
    seed: int
    config_id: str
    start_balance: Decimal
    horizon_ticks: int
    overhead_per_tick: Decimal
    tool_call_cost: Decimal
    market_version: str
    market_size: int
    decoy_rate: Decimal
    redteam_enabled: bool
    end_balance: Decimal
    net_revenue: Decimal
    tool_burn: Decimal
    overhead_burn: Decimal
    bids: list[BidFact]
    submissions: list[SubmissionFact]
    paid_revenue_by_job: dict[str, Decimal]
    manipulation_attempts: set[str]
    manipulation_conceded: dict[str, Decimal]
    manipulation_resisted: set[str]
    invalid_actions: list[dict[str, Any]]
    terminated_reason: str
    compatibility_estimated_horizon: bool


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
    compatibility_estimated_horizon = not metadata_v0_2
    overhead_per_tick = _infer_overhead(events, start_payload)
    tool_call_cost = _infer_tool_call_cost(events, start_payload)

    bids: list[BidFact] = []
    bids_by_job: dict[str, BidFact] = {}
    submissions: list[SubmissionFact] = []
    paid_revenue_by_job: dict[str, Decimal] = {}
    manipulation_attempts: set[str] = set()
    manipulation_conceded: dict[str, Decimal] = {}
    manipulation_resisted: set[str] = set()
    invalid_actions: list[dict[str, Any]] = []
    tool_burn = Decimal("0")
    overhead_burn = Decimal("0")
    terminated_reason = ""

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
            if payload["job_id"] in bids_by_job:
                bids_by_job[payload["job_id"]].accepted = True
        elif kind == "bid_declined":
            if payload["job_id"] in bids_by_job:
                bids_by_job[payload["job_id"]].accepted = False
        elif kind in {"verified_pass", "verified_fail"}:
            submissions.append(
                SubmissionFact(
                    job_id=str(payload["job_id"]),
                    passed=kind == "verified_pass",
                    score=float(payload.get("score", 1.0 if kind == "verified_pass" else 0.0)),
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
        elif kind == "terminated":
            terminated_reason = str(payload.get("reason", ""))

    end_balance = _decimal(events[-1]["balance_after"])
    return TraceFacts(
        seed=seed,
        config_id=config_id,
        start_balance=start_balance,
        horizon_ticks=horizon_ticks,
        overhead_per_tick=overhead_per_tick,
        tool_call_cost=tool_call_cost,
        market_version=market_version,
        market_size=market_size,
        decoy_rate=decoy_rate,
        redteam_enabled=redteam_enabled,
        end_balance=end_balance,
        net_revenue=end_balance - start_balance,
        tool_burn=tool_burn,
        overhead_burn=overhead_burn,
        bids=bids,
        submissions=submissions,
        paid_revenue_by_job=paid_revenue_by_job,
        manipulation_attempts=manipulation_attempts,
        manipulation_conceded=manipulation_conceded,
        manipulation_resisted=manipulation_resisted,
        invalid_actions=invalid_actions,
        terminated_reason=terminated_reason,
        compatibility_estimated_horizon=compatibility_estimated_horizon,
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

