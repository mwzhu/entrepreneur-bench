from __future__ import annotations

from decimal import Decimal

from solvent.env.env import Environment
from solvent.env.tool_api import ToolAdapter
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
        api = ToolAdapter(env)
        if self.mode == "invalid_loop":
            self._run_invalid_loop(env, api)
            return
        single_job_modes = {"happy_path", "bad_delivery"}
        attempted_single_job = False
        completed: set[str] = set()
        while not env.terminated():
            jobs = api.dispatch({"name": "list_jobs", "arguments": {}})["result"]
            if self.mode in single_job_modes and attempted_single_job:
                jobs = []
            for public_job in jobs:
                if public_job["id"] in completed:
                    continue
                if self.mode == "procedure" and self._looks_like_public_decoy(public_job):
                    continue
                job = api.dispatch({"name": "inspect_job", "arguments": {"job_id": public_job["id"]}})["result"]
                bid_result = api.dispatch(
                    {"name": "bid", "arguments": {"job_id": job["id"], "price": str(self._bid_price(job))}}
                )["result"]
                completed.add(job["id"])
                attempted_single_job = True
                if bid_result["accepted"]:
                    self._handle_manipulation(api, job["id"], bid_result.get("manipulation"))
                    artifact = self._artifact_for(job["inputs"]["csv"])
                    api.dispatch({"name": "submit", "arguments": {"job_id": job["id"], "artifact": artifact}})
                if self.mode in single_job_modes:
                    break
            if not env.terminated():
                api.dispatch({"name": "end_tick", "arguments": {}})

    def _artifact_for(self, raw_csv: str) -> str:
        if self.mode == "bad_delivery":
            return '"unterminated\n'
        return build_clean_csv(raw_csv)

    def _bid_price(self, job: dict) -> Decimal:
        if self.mode == "overprice":
            return Decimal("100000.00")
        return Decimal(str(job.get("starting_price", "0.50")))

    def _handle_manipulation(self, api: ToolAdapter, job_id: str, manipulation: dict | None) -> None:
        if manipulation is None:
            return
        if self.mode == "naive":
            api.dispatch({"name": "respond", "arguments": {"job_id": job_id, "decision": "discount"}})
        else:
            api.dispatch({"name": "respond", "arguments": {"job_id": job_id, "decision": "hold_price"}})

    def _looks_like_public_decoy(self, job: dict) -> bool:
        return "budget is extremely limited" in job["brief"].lower()

    def _run_invalid_loop(self, env: Environment, api: ToolAdapter) -> None:
        while not env.terminated():
            jobs = api.dispatch({"name": "list_jobs", "arguments": {}})["result"]
            if jobs:
                job = api.dispatch({"name": "inspect_job", "arguments": {"job_id": jobs[0]["id"]}})["result"]
                api.dispatch({"name": "bid", "arguments": {"job_id": job["id"], "price": str(job.get("starting_price", "0.50"))}})
                for _ in range(3):
                    api.dispatch({"name": "bid", "arguments": {"job_id": job["id"], "price": "0.01"}})
                return
            if not env.terminated():
                api.dispatch({"name": "end_tick", "arguments": {}})
