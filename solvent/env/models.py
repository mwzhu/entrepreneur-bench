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
    horizon_minutes: int | None = None
    overhead_per_minute: Decimal | None = None
    market_version: str = "data_clean_static_v0_2"
    market_size: int = 5
    arrival_rate_per_day: Decimal = Decimal("1.00")
    decoy_rate: Decimal = Decimal("0.40")
    manipulation_rate: Decimal = Decimal("0.00")
    redteam_enabled: bool = False
    delivery_mode: str = "tool_mediated"
    menu_version: str = "menu_v0_4"
    menu_checksum: str = ""
    task_mix: dict[str, float] = field(default_factory=lambda: {"data_clean": 1.0})
    difficulty_distribution: dict[str, float] = field(default_factory=lambda: {"easy": 1.0})
    seed_split: str = "ad_hoc"
    pricing_table_version: str = "pricing_v0_4"
    brain_model: str = ""
    context_policy: str = "sliding_window"
    ctx_window_tokens: int = 24000
    caching: bool = False
    corpus_schema_version: str = "none"
    menu_schema_version: str = "solvent_delivery_menu_v0_4"
    work_time_enabled: bool = False
    job_ttl_ticks: int | None = None
    job_ttl_minutes: int | None = None
    reputation_enabled: bool = False
    reputation_start: Decimal = Decimal("1.00")
    reputation_pass_delta: Decimal = Decimal("0.10")
    reputation_fail_delta: Decimal = Decimal("-0.20")
    reputation_concede_delta: Decimal = Decimal("-0.10")
    reputation_gate_threshold: Decimal = Decimal("1.00")
    reputation_high_value_cutoff: Decimal = Decimal("1.50")


@dataclass(frozen=True)
class JobPublic:
    id: str
    type: str
    brief: str
    inputs: dict[str, Any]
    arrival_tick: int
    arrival_minute: int | None = None
    expiry_minute: int | None = None


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
    internal_difficulty: str = "easy"
    arrival_minute: int | None = None
    expiry_minute: int | None = None

    def to_public(self) -> JobPublic:
        return JobPublic(
            id=self.id,
            type=self.type,
            brief=self.brief,
            inputs=dict(self.inputs),
            arrival_tick=self.arrival_tick,
            arrival_minute=self.arrival_minute,
            expiry_minute=self.expiry_minute,
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
    delivery_model: str | None = None
    delivery_attempts: int = 0


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
