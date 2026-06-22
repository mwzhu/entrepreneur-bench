from __future__ import annotations

import random
from dataclasses import replace
from decimal import Decimal

from solvent.env.errors import UnknownJobError
from solvent.env.models import Job, ManipulationEvent
from solvent.tasks.data_clean import generate_data_clean_job
from solvent.tasks.extract import generate_extract_job


class Market:
    """Seeded static job board with frozen v0.1 and scored v0.2 variants."""

    def __init__(
        self,
        seed: int,
        version: str = "data_clean_static_v0_2",
        market_size: int = 5,
        horizon_minutes: int | None = None,
        arrival_rate_per_day: Decimal = Decimal("1.00"),
        job_ttl_minutes: int | None = None,
        decoy_rate: Decimal = Decimal("0.40"),
        redteam_enabled: bool = False,
        task_mix: dict[str, float] | None = None,
        difficulty_distribution: dict[str, float] | None = None,
    ):
        self.seed = seed
        self.version = version
        self.market_size = market_size
        self.horizon_minutes = horizon_minutes
        self.arrival_rate_per_day = arrival_rate_per_day
        self.job_ttl_minutes = job_ttl_minutes
        self.decoy_rate = decoy_rate
        self.redteam_enabled = redteam_enabled
        self.task_mix = task_mix or {"data_clean": 1.0}
        self.difficulty_distribution = difficulty_distribution or {"easy": 1.0}
        if version == "data_clean_static_v0_1":
            self._jobs = [generate_data_clean_job(seed, index) for index in range(3)]
        elif version == "data_clean_static_v0_2":
            self._jobs = self._generate_v0_2_jobs()
        elif version == "business_stream_v0_5":
            self._jobs = self._generate_stream_jobs()
        else:
            raise ValueError(f"unknown market version: {version}")
        self._jobs_by_id = {job.id: job for job in self._jobs}

    def available_jobs(self, tick: int, business_time: bool = False) -> list[Job]:
        if business_time:
            return [
                job
                for job in self._jobs
                if (job.arrival_minute if job.arrival_minute is not None else job.arrival_tick) <= tick
                and (job.expiry_minute is None or tick < job.expiry_minute)
            ]
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
            task_type = self._sample_weighted(self.task_mix, f"{self.seed}:{index}:task-type")
            difficulty = self._sample_weighted(self.difficulty_distribution, f"{self.seed}:{index}:difficulty")
            base = self._generate_job(task_type, index, difficulty)
            is_decoy = index in decoy_indices
            rng = random.Random(f"{self.seed}:{index}:market-v0.2")
            if is_decoy:
                reservation_price = (Decimal(rng.randrange(50, 151)) / Decimal("100")).quantize(Decimal("0.01"))
                est_cost = (Decimal(rng.randrange(6000, 9000)) / Decimal("100")).quantize(Decimal("0.01"))
                brief = base.brief + " The client notes that the budget is extremely limited."
            else:
                reservation_price = (Decimal(rng.randrange(5000, 50001)) / Decimal("100")).quantize(Decimal("0.01"))
                est_cost = (reservation_price * Decimal(rng.randrange(10, 21)) / Decimal("100")).quantize(Decimal("0.01"))
                brief = base.brief
            starting_price = (
                reservation_price * (Decimal(100 - rng.randrange(10, 41)) / Decimal("100"))
            ).quantize(Decimal("0.01"))
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
                    starting_price=starting_price,
                    is_decoy=is_decoy,
                    manipulation=manipulation,
                )
            )
        return jobs

    def _generate_stream_jobs(self) -> list[Job]:
        horizon = self.horizon_minutes or max(1, self.market_size)
        rng = random.Random(f"{self.seed}:stream-arrivals")
        lam = float(self.arrival_rate_per_day) / 1440
        arrivals: list[int] = []
        t = 0.0
        while True:
            t += rng.expovariate(lam)
            if t >= horizon:
                break
            arrivals.append(int(t))
        if not arrivals:
            arrivals = [0]
        self.market_size = len(arrivals)
        ttl = self.job_ttl_minutes if self.job_ttl_minutes is not None else max(60, min(1440, horizon))
        jobs = []
        for arrival, job in zip(sorted(arrivals), self._generate_v0_2_jobs()):
            expiry = min(horizon, arrival + ttl)
            jobs.append(
                replace(
                    job,
                    arrival_tick=arrival,
                    arrival_minute=arrival,
                    expiry_minute=expiry,
                )
            )
        return jobs

    def _generate_job(self, task_type: str, index: int, difficulty: str) -> Job:
        if task_type == "data_clean":
            return generate_data_clean_job(self.seed, index, difficulty)
        if task_type == "extract":
            return generate_extract_job(self.seed, index, difficulty)
        raise ValueError(f"unknown task type: {task_type}")

    def _sample_weighted(self, weights: dict[str, float], key: str) -> str:
        if not weights:
            raise ValueError("weighted distribution cannot be empty")
        rng = random.Random(key)
        total = sum(float(weight) for weight in weights.values())
        if total <= 0:
            raise ValueError("weighted distribution must have positive mass")
        draw = rng.random() * total
        cumulative = 0.0
        for value, weight in sorted(weights.items()):
            cumulative += float(weight)
            if draw <= cumulative:
                return value
        return sorted(weights)[-1]

    def _decoy_indices(self) -> set[int]:
        if self.market_size <= 0:
            return set()
        decoy_count = int((Decimal(self.market_size) * self.decoy_rate).to_integral_value())
        decoy_count = max(0, min(self.market_size, decoy_count))
        if decoy_count == self.market_size:
            return set(range(self.market_size))
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
