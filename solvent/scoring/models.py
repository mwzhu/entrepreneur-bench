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
class ToolSelectionSignal:
    attempted_jobs: int
    tool_mediated_jobs: int
    tool_price_charged: Decimal
    oracle_tool_regret: Decimal
    average_expected_value_ratio: float | None


@dataclass(frozen=True)
class ComputeEconomy:
    brain_tokens_in: int
    brain_tokens_out: int
    brain_cost: Decimal
    fraction_of_optimal_per_compute_dollar: float | None
    brain_cache_read_tokens: int = 0
    brain_cache_write_tokens: int = 0


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
    omniscient_reference_relaxation: bool
    realizable_reference_relaxation: bool
    fraction_of_omniscient_optimal: float | None
    fraction_of_realizable: float | None
    selection: SelectionSignal
    pricing: PricingSignal
    delivery: DeliverySignal
    support: SupportSignal
    coherence: CoherenceSignal
    tool_selection: ToolSelectionSignal | None = None
    compute: ComputeEconomy | None = None
    delivery_mode: str = "direct"
    menu_version: str = "menu_v0_4"
    menu_checksum: str = ""
    seed_split: str = "ad_hoc"


@dataclass(frozen=True)
class MetricSummary:
    mean: float | None
    std: float | None
    n: int
