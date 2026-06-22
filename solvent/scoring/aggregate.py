from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import Iterable


@dataclass(frozen=True)
class DistributionSummary:
    mean: float | None
    std: float | None
    min: float | None
    n: int
    ci95_low: float | None
    ci95_high: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "n": self.n,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
        }


def summarize_distribution(values: Iterable[float | int | None]) -> DistributionSummary:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return DistributionSummary(mean=None, std=None, min=None, n=0, ci95_low=None, ci95_high=None)
    avg = mean(clean)
    sample_std = stdev(clean) if len(clean) > 1 else 0.0
    margin = 1.96 * sample_std / sqrt(len(clean)) if len(clean) > 1 else None
    return DistributionSummary(
        mean=avg,
        std=sample_std,
        min=min(clean),
        n=len(clean),
        ci95_low=avg - margin if margin is not None else None,
        ci95_high=avg + margin if margin is not None else None,
    )
