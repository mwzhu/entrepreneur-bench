from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class SelectionSignal:
    chosen_jobs: int
    good_chosen: int
    decoys_chosen: int
    good_available: int
    precision: float | None
    recall: float | None
    selection_regret: Decimal


@dataclass(frozen=True)
class PricingSignal:
    accepted_jobs: int
    declined_good_jobs: int
    average_price_ratio: float | None
    surplus_left: Decimal
    lost_to_overprice: Decimal
    pricing_regret: Decimal


@dataclass(frozen=True)
class DeliverySignal:
    submitted_jobs: int
    passed_jobs: int
    pass_rate: float | None
    average_verify_score: float | None


@dataclass(frozen=True)
class SupportSignal:
    manipulation_attempts: int
    manipulation_resisted: int
    manipulation_conceded: int
    conceded_value: Decimal
    resistance_rate: float | None


@dataclass(frozen=True)
class CoherenceSignal:
    dropped_jobs: int
    duplicate_bid_attempts: int
    invalid_actions: int
    undelivered_in_progress: int
    action_loops: int
    coherence_penalty: Decimal


@dataclass(frozen=True)
class Scorecard:
    seed: int
    config_id: str
    trace_path: Path
    compatibility_estimated_horizon: bool
    net_revenue: Decimal
    gross_score: Decimal
    omniscient_optimal_net: Decimal
    realizable_reference_net: Decimal
    fraction_of_omniscient_optimal: float | None
    fraction_of_realizable: float | None
    selection: SelectionSignal
    pricing: PricingSignal
    delivery: DeliverySignal
    support: SupportSignal
    coherence: CoherenceSignal


@dataclass(frozen=True)
class MetricSummary:
    mean: float | None
    std: float | None
    n: int
