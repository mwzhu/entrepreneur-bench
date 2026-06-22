from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

from solvent.scoring.models import MetricSummary, Scorecard


@dataclass(frozen=True)
class ConfigSummary:
    net_revenue: MetricSummary
    fraction_of_omniscient_optimal: MetricSummary
    delivery_pass_rate: MetricSummary
    brain_compute_cost: MetricSummary
    tool_selection_regret: MetricSummary
    pricing_regret: MetricSummary
    selection_regret: MetricSummary
    support_conceded_value: MetricSummary
    coherence_penalty: MetricSummary
    manipulation_resistance_loss: MetricSummary | None = None


def summarize_scorecards(scorecards: list[Scorecard]) -> ConfigSummary:
    return ConfigSummary(
        net_revenue=_summary([float(card.net_revenue) for card in scorecards]),
        fraction_of_omniscient_optimal=_summary(
            [card.fraction_of_omniscient_optimal for card in scorecards if card.fraction_of_omniscient_optimal is not None]
        ),
        delivery_pass_rate=_summary([card.delivery.pass_rate for card in scorecards if card.delivery.pass_rate is not None]),
        brain_compute_cost=_summary([float(card.compute.brain_cost) for card in scorecards if card.compute is not None]),
        tool_selection_regret=_summary(
            [float(card.tool_selection.oracle_tool_regret) for card in scorecards if card.tool_selection is not None]
        ),
        pricing_regret=_summary([float(card.pricing.pricing_regret) for card in scorecards]),
        selection_regret=_summary([float(card.selection.selection_regret) for card in scorecards]),
        support_conceded_value=_summary([float(card.support.conceded_value) for card in scorecards]),
        coherence_penalty=_summary([float(card.coherence.coherence_penalty) for card in scorecards]),
    )


def with_manipulation_loss(summary: ConfigSummary, losses: list[float]) -> ConfigSummary:
    return ConfigSummary(
        net_revenue=summary.net_revenue,
        fraction_of_omniscient_optimal=summary.fraction_of_omniscient_optimal,
        delivery_pass_rate=summary.delivery_pass_rate,
        brain_compute_cost=summary.brain_compute_cost,
        tool_selection_regret=summary.tool_selection_regret,
        pricing_regret=summary.pricing_regret,
        selection_regret=summary.selection_regret,
        support_conceded_value=summary.support_conceded_value,
        coherence_penalty=summary.coherence_penalty,
        manipulation_resistance_loss=_summary(losses),
    )


def paired_delta_scorecards(a_cards: list[Scorecard], b_cards: list[Scorecard]) -> ConfigSummary:
    pairs = _paired_scorecards(a_cards, b_cards)
    return ConfigSummary(
        net_revenue=_summary([float(b.net_revenue - a.net_revenue) for a, b in pairs]),
        fraction_of_omniscient_optimal=_summary(
            [
                b.fraction_of_omniscient_optimal - a.fraction_of_omniscient_optimal
                for a, b in pairs
                if b.fraction_of_omniscient_optimal is not None and a.fraction_of_omniscient_optimal is not None
            ]
        ),
        delivery_pass_rate=_summary(
            [
                b.delivery.pass_rate - a.delivery.pass_rate
                for a, b in pairs
                if b.delivery.pass_rate is not None and a.delivery.pass_rate is not None
            ]
        ),
        brain_compute_cost=_summary(
            [
                float(b.compute.brain_cost - a.compute.brain_cost)
                for a, b in pairs
                if b.compute is not None and a.compute is not None
            ]
        ),
        tool_selection_regret=_summary(
            [
                float(b.tool_selection.oracle_tool_regret - a.tool_selection.oracle_tool_regret)
                for a, b in pairs
                if b.tool_selection is not None and a.tool_selection is not None
            ]
        ),
        pricing_regret=_summary([float(b.pricing.pricing_regret - a.pricing.pricing_regret) for a, b in pairs]),
        selection_regret=_summary([float(b.selection.selection_regret - a.selection.selection_regret) for a, b in pairs]),
        support_conceded_value=_summary([float(b.support.conceded_value - a.support.conceded_value) for a, b in pairs]),
        coherence_penalty=_summary([float(b.coherence.coherence_penalty - a.coherence.coherence_penalty) for a, b in pairs]),
    )


def paired_delta_values(a_values: dict[int, float], b_values: dict[int, float]) -> MetricSummary:
    seeds = sorted(set(a_values) & set(b_values))
    return _summary([b_values[seed] - a_values[seed] for seed in seeds])


def _summary(values: list[float | None]) -> MetricSummary:
    clean = [value for value in values if value is not None]
    if not clean:
        return MetricSummary(mean=None, std=None, n=0)
    return MetricSummary(mean=mean(clean), std=stdev(clean) if len(clean) > 1 else 0.0, n=len(clean))


def _paired_scorecards(a_cards: list[Scorecard], b_cards: list[Scorecard]) -> list[tuple[Scorecard, Scorecard]]:
    a_index = _cards_by_seed_occurrence(a_cards)
    b_index = _cards_by_seed_occurrence(b_cards)
    return [
        (a_index[key], b_index[key])
        for key in sorted(set(a_index) & set(b_index))
    ]


def _cards_by_seed_occurrence(cards: list[Scorecard]) -> dict[tuple[int, int], Scorecard]:
    counts: dict[int, int] = {}
    indexed = {}
    for card in cards:
        occurrence = counts.get(card.seed, 0)
        counts[card.seed] = occurrence + 1
        indexed[(card.seed, occurrence)] = card
    return indexed
