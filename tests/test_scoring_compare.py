from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from solvent.scoring.compare import paired_delta_scorecards, paired_delta_values, summarize_scorecards
from solvent.scoring.models import (
    CoherenceSignal,
    DeliverySignal,
    PricingSignal,
    Scorecard,
    SelectionSignal,
    SupportSignal,
)


def test_summary_mean_and_sample_std() -> None:
    summary = summarize_scorecards([_card(1, net="1.00"), _card(2, net="3.00")])
    assert summary.net_revenue.mean == 2.0
    assert round(summary.net_revenue.std, 6) == 1.414214


def test_paired_delta_aligns_by_seed_not_position() -> None:
    a_cards = [_card(2, net="2.00", fraction=0.2), _card(1, net="1.00", fraction=0.1)]
    b_cards = [_card(1, net="4.00", fraction=0.4), _card(2, net="8.00", fraction=0.8)]
    delta = paired_delta_scorecards(a_cards, b_cards)
    assert delta.net_revenue.mean == 4.5
    assert round(delta.fraction_of_omniscient_optimal.mean, 6) == 0.45


def test_missing_paired_values_are_skipped() -> None:
    delta = paired_delta_values({1: 0.4, 2: 0.5}, {2: 0.2, 3: 0.9})
    assert delta.n == 1
    assert delta.mean == -0.3


def test_compare_summary_is_stable_dict_shape() -> None:
    summary = summarize_scorecards([_card(1, net="1.00")])
    assert sorted(summary.__dict__) == [
        "delivery_pass_rate",
        "fraction_of_omniscient_optimal",
        "manipulation_resistance_loss",
        "net_revenue",
        "pricing_regret",
        "selection_regret",
    ]


def _card(seed: int, net: str, fraction: float | None = 0.5) -> Scorecard:
    return Scorecard(
        seed=seed,
        config_id="stub:test",
        trace_path=Path(f"/tmp/{seed}.jsonl"),
        compatibility_estimated_horizon=False,
        net_revenue=Decimal(net),
        gross_score=Decimal("1.00"),
        omniscient_optimal_net=Decimal("10.00"),
        realizable_reference_net=Decimal("10.00"),
        fraction_of_omniscient_optimal=fraction,
        fraction_of_realizable=fraction,
        selection=SelectionSignal(1, 1, 0, 1, 1.0, 1.0, Decimal("0.00")),
        pricing=PricingSignal(1, 0, 1.0, Decimal("0.00"), Decimal("0.00"), Decimal("0.00")),
        delivery=DeliverySignal(1, 1, 1.0, 1.0),
        support=SupportSignal(0, 0, 0, Decimal("0.00"), None),
        coherence=CoherenceSignal(0, 0, 0, 0, 0, Decimal("0.00")),
    )
