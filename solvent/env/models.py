from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class EnvConfig:
    seed: int
    config_id: str
    start_balance: Decimal
    horizon_ticks: int
    overhead_per_tick: Decimal
    tool_call_cost: Decimal
    trace_path: Path
    market_version: str = "data_clean_static_v0_2"
    market_size: int = 5
    decoy_rate: Decimal = Decimal("0.40")
    redteam_enabled: bool = False


@dataclass(frozen=True)
class JobPublic:
    id: str
    type: str
    brief: str
    inputs: dict[str, Any]
    arrival_tick: int


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class Check:
    name: str
    predicate: Callable[[str, dict[str, Any]], CheckResult]


@dataclass(frozen=True)
class Rubric:
    checks: tuple[Check, ...]


@dataclass(frozen=True)
class Job:
    id: str
    type: str
    brief: str
    inputs: dict[str, Any]
    arrival_tick: int
    reservation_price: Decimal
    est_cost: Decimal
    rubric: Rubric
    is_decoy: bool = False
    manipulation: ManipulationEvent | None = None

    def to_public(self) -> JobPublic:
        return JobPublic(
            id=self.id,
            type=self.type,
            brief=self.brief,
            inputs=dict(self.inputs),
            arrival_tick=self.arrival_tick,
        )

    @property
    def true_value(self) -> Decimal:
        return self.reservation_price - self.est_cost


@dataclass(frozen=True)
class ManipulationEvent:
    kind: str
    message: str
    discount_fraction: Decimal

    def to_public(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "discount_fraction": self.discount_fraction,
        }


@dataclass
class PendingManipulation:
    job_id: str
    event: ManipulationEvent
    resolved: bool = False


@dataclass(frozen=True)
class VerifyResult:
    checks: list[CheckResult]
    passed: bool = field(init=False)
    score: float = field(init=False)

    def __post_init__(self) -> None:
        total = len(self.checks)
        passed_count = sum(1 for check in self.checks if check.passed)
        object.__setattr__(self, "passed", total > 0 and passed_count == total)
        object.__setattr__(self, "score", passed_count / total if total else 0.0)


@dataclass(frozen=True)
class Event:
    tick: int
    kind: str
    payload: dict[str, Any]
    balance_after: Decimal
    burn_delta: Decimal


@dataclass
class AcceptedJob:
    job_id: str
    contract_price: Decimal
    submitted: bool = False
    paid: bool = False


@dataclass(frozen=True)
class EpisodeSummary:
    seed: int
    config_id: str
    start_balance: Decimal
    end_balance: Decimal
    net_revenue: Decimal
    terminated_reason: str
    ticks_elapsed: int
    jobs_paid: int
    trace_path: Path
