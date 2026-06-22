from decimal import Decimal

from solvent.env.market import Market


def test_business_stream_arrivals_and_expiry_are_seeded_and_reconstructable() -> None:
    first = Market(
        42,
        version="business_stream_v0_5",
        horizon_minutes=1440,
        arrival_rate_per_day=Decimal("3"),
        job_ttl_minutes=120,
        decoy_rate=Decimal("0"),
    ).all_jobs()
    second = Market(
        42,
        version="business_stream_v0_5",
        horizon_minutes=1440,
        arrival_rate_per_day=Decimal("3"),
        job_ttl_minutes=120,
        decoy_rate=Decimal("0"),
    ).all_jobs()

    assert [job.id for job in first] == [job.id for job in second]
    assert [job.arrival_minute for job in first] == [0, 480, 960]
    assert [job.expiry_minute for job in first] == [120, 600, 1080]
    assert [job.arrival_minute for job in first] == [job.arrival_minute for job in second]


def test_business_stream_available_jobs_respects_expiry() -> None:
    market = Market(
        42,
        version="business_stream_v0_5",
        horizon_minutes=1440,
        arrival_rate_per_day=Decimal("2"),
        job_ttl_minutes=100,
        decoy_rate=Decimal("0"),
    )

    assert [job.arrival_minute for job in market.available_jobs(0, business_time=True)] == [0]
    assert market.available_jobs(101, business_time=True) == []
    assert [job.arrival_minute for job in market.available_jobs(720, business_time=True)] == [720]
