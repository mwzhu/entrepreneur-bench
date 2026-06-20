from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from solvent.scoring.events import facts_from_events, load_events
from solvent.scoring.scorecard import score_trace, scorecard_to_dict

EVENT_TITLES = {
    "episode_started": "Episode started",
    "board_seen": "Board seen",
    "inspected": "Job inspected",
    "bid_made": "Bid made",
    "bid_accepted": "Bid accepted",
    "bid_declined": "Bid declined",
    "manipulation_attempt": "Manipulation attempt",
    "manipulation_conceded": "Discount conceded",
    "manipulation_resisted": "Discount resisted",
    "submitted": "Artifact submitted",
    "verified_pass": "Verifier passed",
    "verified_fail": "Verifier failed",
    "paid": "Payment credited",
    "invalid_action": "Invalid action",
    "overhead_charged": "Overhead charged",
    "tick_advanced": "Tick advanced",
    "terminated": "Episode terminated",
}


def build_trace_view(trace_path: Path, scorecard_path: Path | None = None, root_dir: Path | None = None) -> dict[str, Any]:
    events = load_events(trace_path)
    facts = facts_from_events(events)
    scorecard = _load_scorecard(trace_path, scorecard_path)
    base_dir = root_dir or trace_path.parent
    scorecard["trace_path"] = _relative_path(trace_path, base_dir)

    jobs = _extract_jobs(events)
    verifier_by_job = _verifier_by_job(events)
    view_events = []
    for index, event in enumerate(events):
        payload = event.get("payload", {})
        job_id = _job_id(payload)
        verify = None
        if event["kind"] in {"submitted", "verified_pass", "verified_fail"} and job_id is not None:
            verify = verifier_by_job.get(job_id)
        view_events.append(
            {
                "index": index,
                "tick": event["tick"],
                "kind": event["kind"],
                "title": EVENT_TITLES.get(event["kind"], event["kind"].replace("_", " ").title()),
                "summary": _event_summary(event),
                "balance_after": event.get("balance_after", "0"),
                "burn_delta": event.get("burn_delta", "0"),
                "payload": payload,
                "job_id": job_id,
                "artifact_preview": payload.get("artifact_preview"),
                "artifact_size": payload.get("artifact_size"),
                "artifact_sha256": payload.get("artifact_sha256"),
                "artifact_truncated": payload.get("artifact_truncated"),
                "verify": verify,
            }
        )

    return {
        "schema_version": "solvent_trace_view_v0_3",
        "trace_path": _relative_path(trace_path, base_dir),
        "scorecard_path": _relative_path(scorecard_path, base_dir) if scorecard_path is not None else None,
        "seed": facts.seed,
        "config_id": facts.config_id,
        "redteam_enabled": facts.redteam_enabled,
        "terminated_reason": facts.terminated_reason,
        "scorecard": scorecard,
        "balance_curve": [
            {
                "event_index": index,
                "tick": event["tick"],
                "kind": event["kind"],
                "balance": event.get("balance_after", "0"),
            }
            for index, event in enumerate(events)
        ],
        "events": view_events,
        "jobs": jobs,
    }


def _load_scorecard(trace_path: Path, scorecard_path: Path | None) -> dict[str, Any]:
    if scorecard_path is not None and scorecard_path.exists():
        return json.loads(scorecard_path.read_text(encoding="utf-8"))
    return scorecard_to_dict(score_trace(trace_path))


def _extract_jobs(events: list[dict[str, Any]]) -> dict[str, Any]:
    jobs: dict[str, Any] = {}
    for event in events:
        payload = event.get("payload", {})
        if event["kind"] == "board_seen":
            for job in payload.get("jobs", []):
                if "id" in job:
                    jobs[str(job["id"])] = job
        elif event["kind"] == "inspected":
            job = payload.get("job", {})
            if "id" in job:
                jobs[str(job["id"])] = job
    return jobs


def _verifier_by_job(events: list[dict[str, Any]]) -> dict[str, Any]:
    verifiers = {}
    for event in events:
        if event["kind"] not in {"verified_pass", "verified_fail"}:
            continue
        payload = event.get("payload", {})
        job_id = _job_id(payload)
        if job_id is None:
            continue
        verifiers[job_id] = {
            "passed": event["kind"] == "verified_pass",
            "score": payload.get("score"),
            "checks": payload.get("checks", []),
        }
    return verifiers


def _event_summary(event: dict[str, Any]) -> str:
    payload = event.get("payload", {})
    kind = event["kind"]
    if kind == "episode_started":
        return f"{payload.get('config_id')} on seed {payload.get('seed')}"
    if kind == "board_seen":
        return f"{len(payload.get('jobs', []))} public job(s)"
    if kind == "bid_made":
        return f"{payload.get('job_id')} at {payload.get('price')}"
    if kind == "bid_accepted":
        return f"{payload.get('job_id')} accepted at {payload.get('contract_price')}"
    if kind == "bid_declined":
        return f"{payload.get('job_id')} declined at {payload.get('price')}"
    if kind == "paid":
        return f"{payload.get('job_id')} paid {payload.get('revenue')}"
    if kind == "manipulation_attempt":
        return str(payload.get("message", "Manipulation attempt"))
    if kind == "manipulation_conceded":
        return f"{payload.get('job_id')} conceded {payload.get('conceded_value')}"
    if kind == "manipulation_resisted":
        return f"{payload.get('job_id')} resisted"
    if kind in {"verified_pass", "verified_fail"}:
        return f"{payload.get('job_id')} score {payload.get('score')}"
    if kind == "invalid_action":
        return str(payload.get("code", "invalid_action"))
    if kind == "terminated":
        return str(payload.get("reason", "terminated"))
    job_id = _job_id(payload)
    return str(job_id) if job_id is not None else EVENT_TITLES.get(kind, kind)


def _job_id(payload: dict[str, Any]) -> str | None:
    if "job_id" in payload:
        return str(payload["job_id"])
    job = payload.get("job")
    if isinstance(job, dict) and "id" in job:
        return str(job["id"])
    return None


def _relative_path(path: Path | None, root_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
