from __future__ import annotations


class Clock:
    """Deterministic integer clock.

    Legacy episodes use one unit as one tick. v0.5 business-time episodes use
    the same integer counter as minutes; ``tick`` remains an alias so older
    traces and tests stay stable.
    """

    def __init__(self, horizon_ticks: int | None = None, horizon_minutes: int | None = None):
        if horizon_ticks is None and horizon_minutes is None:
            raise ValueError("horizon_ticks or horizon_minutes is required")
        horizon = horizon_minutes if horizon_minutes is not None else horizon_ticks
        if horizon is None or horizon < 0:
            raise ValueError("clock horizon must be non-negative")
        self._horizon = horizon
        self._time = 0

    @property
    def tick(self) -> int:
        return self._time

    @property
    def business_time(self) -> int:
        return self._time

    @property
    def horizon(self) -> int:
        return self._horizon

    def advance(self, amount: int = 1) -> int:
        if amount < 0:
            raise ValueError("clock advance amount must be non-negative")
        self._time = min(self._horizon, self._time + amount)
        return self._time

    def advance_to(self, target: int) -> int:
        if target < self._time:
            return self._time
        return self.advance(target - self._time)

    def reached_horizon(self) -> bool:
        return self._time >= self._horizon
