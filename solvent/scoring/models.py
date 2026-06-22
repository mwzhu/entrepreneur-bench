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
    """Counter-offer pricing quality (v0.6).

    Under the starting-price / one-shot-counter mechanic the optimal play is to
    counter exactly at the hidden reservation, capturing the full surplus. Regret
    is the surplus left on the table on good jobs the agent actually took, whether
    it accepted the floor (forfeiting reservation - starting_price) or countered
    below the ceiling (forfeiting reservation - counter). Walking away from a good
    job after a rejected counter is a selection loss, not a pricing one, so it is
    not counted here.
    """

    accepted_jobs: int
    floor_accepts: int
    counter_accepts: int
    rejected_counters: int
    average_price_ratio: float | None
    surplus_left: Decimal
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
    threshold_policy_reference_net: Decimal = Decimal("0")
    fraction_of_threshold_policy: float | None = None
    # Expected-value (control-variate) scoring — removes delivery-luck variance by
    # crediting contract_price * pass_prob instead of the realized 0/1 (Task 8).
    expected_net_revenue: Decimal = Decimal("0")
    fraction_of_omniscient_optimal_expected: float | None = None
    fraction_of_realizable_expected: float | None = None
    # Approximate joint model x schedule optimum (Task 13); relaxation-labelled.
    joint_optimum_reference_net: Decimal = Decimal("0")
    joint_optimum_reference_relaxation: bool = False
    fraction_of_joint_optimum: float | None = None
    # Realized-horizon scoring — reference optimum over only the jobs reachable before
    # the agent actually stopped, so capability metrics aren't truncation-confounded.
    # Equals the full-horizon optimum when the run reached the horizon.
    horizon_ticks: int = 0
    realized_horizon_ticks: int = 0
    omniscient_optimal_net_realized: Decimal = Decimal("0")
    omniscient_realized_relaxation: bool = False
    fraction_of_omniscient_optimal_realized: float | None = None
    fraction_of_omniscient_optimal_realized_expected: float | None = None


@dataclass(frozen=True)
class MetricSummary:
    mean: float | None
    std: float | None
    n: int
