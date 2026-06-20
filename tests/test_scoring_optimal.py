from decimal import Decimal
from pathlib import Path

from solvent.env.market import Market
from solvent.scoring.events import facts_from_events
from solvent.scoring.optimal import omniscient_reference, reachable_jobs, realizable_reference


def test_omniscient_reference_skips_decoys_and_unreachable_jobs() -> None:
    market = Market(42)
    facts = _facts(horizon_ticks=3)
    reference = omniscient_reference(market.all_jobs(), facts)
    assert [job.id for job in reference.selected_jobs] == ["dc-42-0", "dc-42-2"]
    assert all(job.true_value > 0 for job in reference.selected_jobs)
    assert "dc-42-4" not in [job.id for job in reference.selected_jobs]


def test_reachable_jobs_uses_horizon() -> None:
    jobs = reachable_jobs(Market(42).all_jobs(), 3)
    assert [job.arrival_tick for job in jobs] == [0, 1, 2]


def test_realizable_reference_uses_decimal_capability() -> None:
    market = Market(42)
    reference = realizable_reference(market.all_jobs(), _facts(horizon_ticks=5), 0.5)
    assert isinstance(reference.net, Decimal)


def _facts(horizon_ticks: int):
    return facts_from_events(
        [
            {
                "tick": 0,
                "kind": "episode_started",
                "payload": {
                    "seed": 42,
                    "config_id": "stub:test",
                    "start_balance": "20.00",
                    "horizon_ticks": horizon_ticks,
                    "overhead_per_tick": "0.05",
                    "tool_call_cost": "0.01",
                    "market_version": "data_clean_static_v0_2",
                    "market_size": 5,
                    "decoy_rate": "0.40",
                    "redteam_enabled": False,
                },
                "balance_after": "20.00",
                "burn_delta": "0",
            },
            {
                "tick": horizon_ticks,
                "kind": "terminated",
                "payload": {"reason": "turn_cap"},
                "balance_after": "20.00",
                "burn_delta": "0",
            },
        ]
    )

