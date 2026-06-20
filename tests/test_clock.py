from solvent.env.clock import Clock


def test_clock_starts_at_zero() -> None:
    assert Clock(horizon_ticks=3).tick == 0


def test_clock_advances_deterministically() -> None:
    clock = Clock(horizon_ticks=3)
    assert clock.advance() == 1
    assert clock.advance() == 2


def test_clock_reaches_horizon_at_cap() -> None:
    clock = Clock(horizon_ticks=2)
    assert not clock.reached_horizon()
    clock.advance()
    assert not clock.reached_horizon()
    clock.advance()
    assert clock.reached_horizon()
