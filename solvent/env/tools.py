from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ListJobs:
    pass


@dataclass(frozen=True)
class InspectJob:
    job_id: str


@dataclass(frozen=True)
class Bid:
    job_id: str
    price: Decimal


@dataclass(frozen=True)
class Submit:
    job_id: str
    artifact: str


@dataclass(frozen=True)
class Respond:
    job_id: str
    decision: str


@dataclass(frozen=True)
class CheckBalance:
    pass


@dataclass(frozen=True)
class ListInProgress:
    pass


@dataclass(frozen=True)
class EndTick:
    pass
