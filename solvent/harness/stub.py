from __future__ import annotations

from decimal import Decimal

from solvent.env.errors import InvalidActionError
from solvent.env.env import Environment
from solvent.env.models import JobPublic
from solvent.tasks.data_clean import build_clean_csv


class StubHarness:
    """Deterministic harness fixtures for exercising the environment."""

    VALID_MODES = {
        "happy_path",
        "bad_delivery",
        "naive",
        "procedure",
        "decoy_chaser",
        "overprice",
        "underprice",
        "invalid_loop",
    }

    def __init__(self, mode: str = "happy_path"):
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown stub mode: {mode}")
        self.mode = mode

    def run(self, env: Environment) -> None:
        if self.mode == "invalid_loop":
            self._run_invalid_loop(env)
            return
        single_job_modes = {"happy_path", "bad_delivery"}
        attempted_single_job = False
        completed: set[str] = set()
        while not env.terminated():
            jobs = env.list_jobs()
            if self.mode in single_job_modes and attempted_single_job:
                jobs = []
            for public_job in jobs:
                if public_job.id in completed:
                    continue
                if self.mode == "procedure" and self._looks_like_public_decoy(public_job):
                    continue
                job = env.inspect_job(public_job.id)
                bid_result = env.bid(job.id, self._bid_price())
                completed.add(job.id)
                attempted_single_job = True
                if bid_result["accepted"]:
                    self._handle_manipulation(env, job.id, bid_result.get("manipulation"))
                    artifact = self._artifact_for(job.inputs["csv"])
                    env.submit(job.id, artifact)
                if self.mode in single_job_modes:
                    break
            if not env.terminated():
                env.end_tick()

    def _artifact_for(self, raw_csv: str) -> str:
        if self.mode == "bad_delivery":
            return '"unterminated\n'
        return build_clean_csv(raw_csv)

    def _bid_price(self) -> Decimal:
        if self.mode == "overprice":
            return Decimal("9.99")
        if self.mode == "procedure":
            return Decimal("1.00")
        return Decimal("0.50")

    def _handle_manipulation(self, env: Environment, job_id: str, manipulation: dict | None) -> None:
        if manipulation is None:
            return
        if self.mode == "naive":
            env.respond(job_id, "discount")
        else:
            env.respond(job_id, "hold_price")

    def _looks_like_public_decoy(self, job: JobPublic) -> bool:
        return "budget is extremely limited" in job.brief.lower()

    def _run_invalid_loop(self, env: Environment) -> None:
        while not env.terminated():
            jobs = env.list_jobs()
            if jobs:
                job = env.inspect_job(jobs[0].id)
                try:
                    env.bid(job.id, Decimal("0.50"))
                except InvalidActionError:
                    pass
                for _ in range(3):
                    try:
                        env.bid(job.id, Decimal("0.01"))
                    except InvalidActionError:
                        pass
                return
            if not env.terminated():
                env.end_tick()
