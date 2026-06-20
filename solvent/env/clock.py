class Clock:
    """Deterministic tick counter with a fixed horizon."""

    def __init__(self, horizon_ticks: int):
        if horizon_ticks < 0:
            raise ValueError("horizon_ticks must be non-negative")
        self._horizon_ticks = horizon_ticks
        self._tick = 0

    @property
    def tick(self) -> int:
        return self._tick

    def advance(self) -> int:
        self._tick += 1
        return self._tick

    def reached_horizon(self) -> bool:
        return self._tick >= self._horizon_ticks
