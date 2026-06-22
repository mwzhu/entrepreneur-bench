from decimal import Decimal

from solvent.delivery.menu import DeliveryMenu
from solvent.env.market import Market


def _stream(seed: int, rate: str, ttl: int, horizon: int = 1440):
    return Market(
        seed,
        version="business_stream_v0_5",
        horizon_minutes=horizon,
        arrival_rate_per_day=Decimal(rate),
        job_ttl_minutes=ttl,
        decoy_rate=Decimal("0"),
    )


def test_business_stream_arrivals_are_seeded_and_reproducible() -> None:
    first = _stream(42, "3", 120).all_jobs()
    second = _stream(42, "3", 120).all_jobs()

    assert [job.id for job in first] == [job.id for job in second]
    assert [job.arrival_minute for job in first] == [job.arrival_minute for job in second]
    # Poisson process: arrivals are non-decreasing and clumpy, not evenly spaced.
    arrivals = [job.arrival_minute for job in first]
    assert arrivals == sorted(arrivals)
    assert arrivals == [1210, 1228, 1258]


def test_business_stream_expiry_is_arrival_plus_ttl_clamped_to_horizon() -> None:
    jobs = _stream(42, "3", 120).all_jobs()
    horizon = 1440
    ttl = 120
    for job in jobs:
        assert job.expiry_minute == min(horizon, job.arrival_minute + ttl)
    assert [job.expiry_minute for job in jobs] == [1330, 1348, 1378]


def test_business_stream_arrival_count_is_random_poisson_count() -> None:
    # Same horizon/seed but different rate yields a different (random) count.
    sparse = _stream(7, "8", 100).all_jobs()
    assert len(sparse) == 11
    arrivals = [job.arrival_minute for job in sparse]
    assert arrivals == [14, 323, 441, 442, 658, 816, 1131, 1246, 1254, 1276, 1362]


def test_business_stream_guarantees_at_least_one_job() -> None:
    # Tiny horizon / low rate can draw zero Poisson arrivals; force one at minute 0.
    jobs = _stream(42, "2", 100).all_jobs()
    assert len(jobs) >= 1
    assert jobs[0].arrival_minute == 0


def test_business_stream_available_jobs_respects_arrival_and_expiry() -> None:
    market = _stream(7, "8", 100)
    # First job arrives at minute 14, expires at 114.
    assert market.available_jobs(13, business_time=True) == []
    assert [job.arrival_minute for job in market.available_jobs(14, business_time=True)] == [14]
    # At minute 114 the first job's expiry is reached and it leaves the board.
    assert 14 not in [job.arrival_minute for job in market.available_jobs(114, business_time=True)]


def test_decoys_are_negative_expected_value_with_delivery_tools() -> None:
    market = Market(
        42,
        version="business_stream_v0_5",
        horizon_minutes=1440,
        arrival_rate_per_day=Decimal("8"),
        job_ttl_minutes=120,
        decoy_rate=Decimal("1"),
        difficulty_distribution={"easy": 1.0},
    )
    menu = DeliveryMenu.load_default()

    assert market.all_jobs()
    for job in market.all_jobs():
        assert job.is_decoy
        best_value = max(
            job.reservation_price * Decimal(str(menu.pass_prob(job.type, model.name, job.internal_difficulty))) - model.price
            for model in menu.public_models()
        )
        assert best_value < 0
