from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

from solvent.scoring.models import MetricSummary, Scorecard


@dataclass(frozen=True)
class ConfigSummary:
    net_revenue: MetricSummary
    fraction_of_omniscient_optimal: MetricSummary
    delivery_pass_rate: MetricSummary
    pricing_regret: MetricSummary
    selection_regret: MetricSummary
    manipulation_resistance_loss: MetricSummary | None = None


def summarize_scorecards(scorecards: list[Scorecard]) -> ConfigSummary:
    return ConfigSummary(
        net_revenue=_summary([float(card.net_revenue) for card in scorecards]),
        fraction_of_omniscient_optimal=_summary(
            [card.fraction_of_omniscient_optimal for card in scorecards if card.fraction_of_omniscient_optimal is not None]
        ),
        delivery_pass_rate=_summary([card.delivery.pass_rate for card in scorecards if card.delivery.pass_rate is not None]),
        pricing_regret=_summary([float(card.pricing.pricing_regret) for card in scorecards]),
        selection_regret=_summary([float(card.selection.selection_regret) for card in scorecards]),
    )


def with_manipulation_loss(summary: ConfigSummary, losses: list[float]) -> ConfigSummary:
    return ConfigSummary(
        net_revenue=summary.net_revenue,
        fraction_of_omniscient_optimal=summary.fraction_of_omniscient_optimal,
        delivery_pass_rate=summary.delivery_pass_rate,
        pricing_regret=summary.pricing_regret,
        selection_regret=summary.selection_regret,
        manipulation_resistance_loss=_summary(losses),
    )


def paired_delta_scorecards(a_cards: list[Scorecard], b_cards: list[Scorecard]) -> ConfigSummary:
    a_by_seed = {card.seed: card for card in a_cards}
    b_by_seed = {card.seed: card for card in b_cards}
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    return ConfigSummary(
        net_revenue=_summary([float(b_by_seed[seed].net_revenue - a_by_seed[seed].net_revenue) for seed in seeds]),
        fraction_of_omniscient_optimal=_summary(
            [
                b_by_seed[seed].fraction_of_omniscient_optimal - a_by_seed[seed].fraction_of_omniscient_optimal
                for seed in seeds
                if b_by_seed[seed].fraction_of_omniscient_optimal is not None
                and a_by_seed[seed].fraction_of_omniscient_optimal is not None
            ]
        ),
        delivery_pass_rate=_summary(
            [
                b_by_seed[seed].delivery.pass_rate - a_by_seed[seed].delivery.pass_rate
                for seed in seeds
                if b_by_seed[seed].delivery.pass_rate is not None and a_by_seed[seed].delivery.pass_rate is not None
            ]
        ),
        pricing_regret=_summary([float(b_by_seed[seed].pricing.pricing_regret - a_by_seed[seed].pricing.pricing_regret) for seed in seeds]),
        selection_regret=_summary(
            [float(b_by_seed[seed].selection.selection_regret - a_by_seed[seed].selection.selection_regret) for seed in seeds]
        ),
    )


def paired_delta_values(a_values: dict[int, float], b_values: dict[int, float]) -> MetricSummary:
    seeds = sorted(set(a_values) & set(b_values))
    return _summary([b_values[seed] - a_values[seed] for seed in seeds])


def _summary(values: list[float | None]) -> MetricSummary:
    clean = [value for value in values if value is not None]
    if not clean:
        return MetricSummary(mean=None, std=None, n=0)
    return MetricSummary(mean=mean(clean), std=stdev(clean) if len(clean) > 1 else 0.0, n=len(clean))
