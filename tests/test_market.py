from decimal import Decimal

import pytest

from solvent.env.errors import UnknownJobError
from solvent.env.market import Market


def test_market_generation_is_stable_for_same_seed() -> None:
    first = Market(42).available_jobs(2)
    second = Market(42).available_jobs(2)
    assert [job.id for job in first] == [job.id for job in second]
    assert [job.inputs for job in first] == [job.inputs for job in second]


def test_market_generation_changes_for_different_seeds() -> None:
    first = Market(42).available_jobs(2)
    second = Market(43).available_jobs(2)
    assert [job.id for job in first] != [job.id for job in second]


def test_market_arrival_filtering_is_cumulative() -> None:
    market = Market(42)
    assert [job.arrival_tick for job in market.available_jobs(0)] == [0]
    assert [job.arrival_tick for job in market.available_jobs(1)] == [0, 1]
    assert [job.arrival_tick for job in market.available_jobs(2)] == [0, 1, 2]


def test_market_unknown_job_raises_typed_error() -> None:
    with pytest.raises(UnknownJobError):
        Market(42).get_job("missing")


def test_public_job_excludes_hidden_fields() -> None:
    public = Market(42).available_jobs(0)[0].to_public()
    assert not hasattr(public, "reservation_price")
    assert not hasattr(public, "est_cost")
    assert not hasattr(public, "rubric")
    assert not hasattr(public, "true_value")
    assert not hasattr(public, "is_decoy")


def test_v0_2_market_has_good_jobs_decoys_and_reachable_defaults() -> None:
    market = Market(42)
    jobs = market.all_jobs()
    assert len(jobs) == 5
    assert max(job.arrival_tick for job in jobs) < 5
    assert sum(1 for job in jobs if job.true_value > 0) == 3
    assert sum(1 for job in jobs if job.is_decoy) == 2


def test_redteam_only_changes_manipulation_attachment() -> None:
    off = Market(42, redteam_enabled=False).all_jobs()
    on = Market(42, redteam_enabled=True).all_jobs()
    for off_job, on_job in zip(off, on):
        assert off_job.id == on_job.id
        assert off_job.inputs == on_job.inputs
        assert off_job.reservation_price == on_job.reservation_price
        assert off_job.est_cost == on_job.est_cost
        assert off_job.is_decoy == on_job.is_decoy
        assert off_job.arrival_tick == on_job.arrival_tick
    assert sum(1 for job in off if job.manipulation is not None) == 0
    assert sum(1 for job in on if job.manipulation is not None) == 1


def test_v0_1_compatibility_market_is_frozen_for_seed_42() -> None:
    jobs = Market(42, version="data_clean_static_v0_1").all_jobs()
    assert [job.id for job in jobs] == ["dc-42-0", "dc-42-1", "dc-42-2"]
    assert [job.arrival_tick for job in jobs] == [0, 1, 2]
    assert [job.reservation_price for job in jobs] == [Decimal("1.16"), Decimal("1.11"), Decimal("1.43")]
    assert [job.est_cost for job in jobs] == [Decimal("0.20"), Decimal("0.20"), Decimal("0.20")]
    assert not any(job.is_decoy for job in jobs)
