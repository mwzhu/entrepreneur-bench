from __future__ import annotations

import random
from dataclasses import replace
from decimal import Decimal

from solvent.env.errors import UnknownJobError
from solvent.env.models import Job, ManipulationEvent
from solvent.tasks.data_clean import generate_data_clean_job


class Market:
    """Seeded static job board with frozen v0.1 and scored v0.2 variants."""

    def __init__(
        self,
        seed: int,
        version: str = "data_clean_static_v0_2",
        market_size: int = 5,
        decoy_rate: Decimal = Decimal("0.40"),
        redteam_enabled: bool = False,
    ):
        self.seed = seed
        self.version = version
        self.market_size = market_size
        self.decoy_rate = decoy_rate
        self.redteam_enabled = redteam_enabled
        if version == "data_clean_static_v0_1":
            self._jobs = [generate_data_clean_job(seed, index) for index in range(3)]
        elif version == "data_clean_static_v0_2":
            self._jobs = self._generate_v0_2_jobs()
        else:
            raise ValueError(f"unknown market version: {version}")
        self._jobs_by_id = {job.id: job for job in self._jobs}

    def available_jobs(self, tick: int) -> list[Job]:
        return [job for job in self._jobs if job.arrival_tick <= tick]

    def all_jobs(self) -> list[Job]:
        return list(self._jobs)

    def get_job(self, job_id: str) -> Job:
        try:
            return self._jobs_by_id[job_id]
        except KeyError as exc:
            raise UnknownJobError(f"unknown job id: {job_id}") from exc

    def _generate_v0_2_jobs(self) -> list[Job]:
        decoy_indices = self._decoy_indices()
        manipulation_index = self._manipulation_index(decoy_indices)
        jobs = []
        for index in range(self.market_size):
            base = generate_data_clean_job(self.seed, index)
            is_decoy = index in decoy_indices
            rng = random.Random(f"{self.seed}:{index}:market-v0.2")
            if is_decoy:
                reservation_price = (Decimal(rng.randrange(20, 51)) / Decimal("100")).quantize(Decimal("0.01"))
                est_cost = (Decimal(rng.randrange(60, 91)) / Decimal("100")).quantize(Decimal("0.01"))
                brief = base.brief + " The client notes that the budget is extremely limited."
            else:
                reservation_price = (Decimal(rng.randrange(100, 181)) / Decimal("100")).quantize(Decimal("0.01"))
                est_cost = (Decimal(rng.randrange(20, 51)) / Decimal("100")).quantize(Decimal("0.01"))
                brief = base.brief
            manipulation = None
            if self.redteam_enabled and index == manipulation_index:
                manipulation = ManipulationEvent(
                    kind="discount_bait",
                    message="Any chance you can do 30% off for this one?",
                    discount_fraction=Decimal("0.30"),
                )
            jobs.append(
                replace(
                    base,
                    brief=brief,
                    reservation_price=reservation_price,
                    est_cost=est_cost,
                    is_decoy=is_decoy,
                    manipulation=manipulation,
                )
            )
        return jobs

    def _decoy_indices(self) -> set[int]:
        if self.market_size <= 0:
            return set()
        decoy_count = int((Decimal(self.market_size) * self.decoy_rate).to_integral_value())
        decoy_count = max(0, min(self.market_size, decoy_count))
        if self.market_size == 5 and decoy_count == 2:
            return {1, 3}
        if decoy_count == 0:
            return set()
        step = self.market_size / decoy_count
        return {min(self.market_size - 1, int(round((i + Decimal("0.5")) * Decimal(str(step))))) for i in range(decoy_count)}

    def _manipulation_index(self, decoy_indices: set[int]) -> int | None:
        good_indices = [index for index in range(self.market_size) if index not in decoy_indices]
        if not good_indices:
            return None
        rng = random.Random(f"{self.seed}:manipulation:v0.2")
        return rng.choice(good_indices)
