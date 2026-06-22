from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent.delivery.menu import DeliveryMenu, delivery_draw_key
from solvent.env.market import Market
from solvent.scoring.events import TraceFacts, facts_from_events, load_events
from solvent.scoring.scorecard import score_trace, scorecard_to_dict

EVENT_TITLES = {
    "episode_started": "Episode started",
    "board_seen": "Board seen",
    "inspected": "Job inspected",
    "clarified": "Clarification answered",
    "bid_made": "Bid made",
    "bid_accepted": "Bid accepted",
    "bid_declined": "Bid declined",
    "manipulation_attempt": "Manipulation attempt",
    "manipulation_conceded": "Discount conceded",
    "manipulation_resisted": "Discount resisted",
    "submitted": "Artifact submitted",
    "verified_pass": "Verifier passed",
    "verified_fail": "Verifier failed",
    "models_listed": "Delivery models listed",
    "delivered": "Delivery attempted",
    "tool_price_charged": "Delivery tool charged",
    "delivery_passed": "Delivery passed",
    "delivery_failed": "Delivery failed",
    "brain_metered": "Brain compute metered",
    "paid": "Payment credited",
    "invalid_action": "Invalid action",
    "overhead_charged": "Overhead charged",
    "tick_advanced": "Tick advanced",
    "business_time_advanced": "Business time advanced",
    "terminated": "Episode terminated",
}

# Per-turn compute accounting clutters the human timeline (one row before every
# action). It is hidden from the rendered timeline but still drives turn
# segmentation and the scorecard's brain-cost figures.
HIDDEN_TIMELINE_KINDS = {"brain_metered"}

# Delivery events that carry a reconstructed ground-truth diagnosis.
_DELIVERY_KINDS = {"delivered", "delivery_passed", "delivery_failed"}


def build_trace_view(trace_path: Path, scorecard_path: Path | None = None, root_dir: Path | None = None) -> dict[str, Any]:
    events = load_events(trace_path)
    facts = facts_from_events(events)
    scorecard = _load_scorecard(trace_path, scorecard_path)
    base_dir = root_dir or trace_path.parent
    scorecard["trace_path"] = _relative_path(trace_path, base_dir)

    jobs = _extract_jobs(events)
    verifier_by_job = _verifier_by_job(events)
    turn_by_index = _assign_turns(events)
    diagnosis_by_key, delivery_summary = _delivery_diagnosis(events, facts)
    turns = _load_agent_turns(trace_path.with_suffix(".llm.jsonl"))

    view_events = []
    for index, event in enumerate(events):
        if event["kind"] in HIDDEN_TIMELINE_KINDS:
            continue
        payload = event.get("payload", {})
        job_id = _job_id(payload)
        verify = None
        if event["kind"] in {"submitted", "verified_pass", "verified_fail"} and job_id is not None:
            verify = verifier_by_job.get(job_id)
        diagnosis = None
        if event["kind"] in _DELIVERY_KINDS:
            diagnosis = diagnosis_by_key.get((job_id, int(payload.get("attempt_index", 0))))
        view_events.append(
            {
                "index": len(view_events),
                "event_index": index,
                "turn": turn_by_index.get(index),
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
                "diagnosis": diagnosis,
            }
        )

    return {
        "schema_version": "solvent_trace_view_v0_5",
        "trace_path": _relative_path(trace_path, base_dir),
        "scorecard_path": _relative_path(scorecard_path, base_dir) if scorecard_path is not None else None,
        "seed": facts.seed,
        "config_id": facts.config_id,
        "redteam_enabled": facts.redteam_enabled,
        "delivery_mode": facts.delivery_mode,
        "terminated_reason": facts.terminated_reason,
        "scorecard": scorecard,
        "balance_curve": [
            {
                "event_index": event["event_index"],
                "tick": event["tick"],
                "kind": event["kind"],
                "balance": event["balance_after"],
            }
            for event in view_events
        ],
        "events": view_events,
        "jobs": jobs,
        "turns": turns,
        "delivery_summary": delivery_summary,
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


def _assign_turns(events: list[dict[str, Any]]) -> dict[int, int | None]:
    """Map each event index to the agent turn that produced it.

    The harness emits a ``brain_metered`` row immediately before dispatching each
    turn's tool call, so those rows delimit turns. Events before the first
    ``brain_metered`` (e.g. ``episode_started``) belong to no turn.
    """
    turn_by_index: dict[int, int | None] = {}
    current_turn = -1
    for index, event in enumerate(events):
        if event["kind"] == "brain_metered":
            current_turn += 1
        turn_by_index[index] = current_turn if current_turn >= 0 else None
    return turn_by_index


def _reconstruct_market(facts: TraceFacts) -> dict[str, Any] | None:
    """Rebuild the seeded job market so the viewer can read hidden ground truth.

    Deterministic from the trace's own provenance; returns ``None`` if the market
    cannot be rebuilt so the viewer still renders without a diagnosis.
    """
    try:
        market = Market(
            seed=facts.seed,
            version=facts.market_version,
            market_size=facts.market_size,
            horizon_minutes=facts.horizon_minutes,
            arrival_rate_per_day=facts.arrival_rate_per_day,
            job_ttl_minutes=facts.job_ttl_minutes,
            decoy_rate=facts.decoy_rate,
            redteam_enabled=facts.redteam_enabled,
            task_mix=facts.task_mix,
            difficulty_distribution=facts.difficulty_distribution,
        )
        return {job.id: job for job in market.all_jobs()}
    except Exception:  # pragma: no cover - viewer must degrade gracefully
        return None


def _delivery_diagnosis(
    events: list[dict[str, Any]], facts: TraceFacts
) -> tuple[dict[tuple[str | None, int], dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct, per tool-mediated delivery, whether a failure was unlucky RNG
    or a sub-optimal model pick — using hidden difficulty + the menu's pass
    probabilities and the same deterministic draw the environment used.
    """
    diagnosis_by_key: dict[tuple[str | None, int], dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []
    try:
        menu = DeliveryMenu.load_default()
    except Exception:  # pragma: no cover
        return diagnosis_by_key, summary

    jobs_by_id: dict[str, Any] | None = None  # rebuilt lazily, only for un-baked traces
    contract_by_job: dict[str, Decimal] = {}
    for event in events:
        kind = event["kind"]
        payload = event.get("payload", {})
        if kind == "bid_accepted":
            contract_by_job[str(payload["job_id"])] = _decimal(payload.get("contract_price", "0"))
        elif kind == "manipulation_conceded":
            contract_by_job[str(payload["job_id"])] = _decimal(payload.get("new_contract_price", "0"))
        elif kind == "delivered":
            job_id = str(payload["job_id"])
            model = str(payload.get("model", ""))
            attempt_index = int(payload.get("attempt_index", 0))
            contract = contract_by_job.get(job_id, Decimal("0"))
            ground_truth = payload.get("ground_truth")
            if ground_truth:
                # Self-contained trace: read the baked answer key directly.
                diagnosis = _diagnosis_from_baked(menu, ground_truth, model, contract)
            else:
                # Older trace: reconstruct difficulty + draw from seed + menu.
                if jobs_by_id is None:
                    jobs_by_id = _reconstruct_market(facts) or {}
                job = jobs_by_id.get(job_id)
                diagnosis = (
                    _diagnose_delivery(menu, facts, job, model, attempt_index, contract)
                    if job is not None
                    else None
                )
            if diagnosis is None:
                continue
            diagnosis_by_key[(job_id, attempt_index)] = diagnosis
            summary.append({"job_id": job_id, "attempt_index": attempt_index, **diagnosis})
    return diagnosis_by_key, summary


def _diagnose_delivery(
    menu: DeliveryMenu,
    facts: TraceFacts,
    job: Any,
    model: str,
    attempt_index: int,
    contract_price: Decimal,
) -> dict[str, Any] | None:
    """Reconstruct a diagnosis for traces written before ground truth was baked in."""
    try:
        draw_key = delivery_draw_key(facts.seed, job.id, model, attempt_index, facts.menu_version)
        resolution = menu.resolve(job.type, model, job.internal_difficulty, draw_key)
        pass_probs = menu.pass_prob_by_model(job.type, job.internal_difficulty)
    except (KeyError, ValueError):
        return None
    return _diagnosis_from_facts(
        menu, model, job.type, job.internal_difficulty, resolution.pass_prob, resolution.draw, pass_probs, contract_price
    )


def _diagnosis_from_baked(
    menu: DeliveryMenu, ground_truth: dict[str, Any], model: str, contract_price: Decimal
) -> dict[str, Any] | None:
    """Read the diagnosis from the ground truth the environment baked into the trace."""
    try:
        pass_probs = {str(name): float(prob) for name, prob in (ground_truth.get("model_pass_probs") or {}).items()}
        pass_prob = float(ground_truth["pass_prob"])
        draw = float(ground_truth["draw"])
    except (TypeError, ValueError, KeyError):
        return None
    return _diagnosis_from_facts(
        menu,
        model,
        ground_truth.get("task_type"),
        ground_truth.get("internal_difficulty"),
        pass_prob,
        draw,
        pass_probs,
        contract_price,
    )


def _diagnosis_from_facts(
    menu: DeliveryMenu,
    model: str,
    task_type: str | None,
    difficulty: str | None,
    pass_prob: float,
    draw: float,
    pass_probs: dict[str, float],
    contract_price: Decimal,
) -> dict[str, Any] | None:
    if model not in pass_probs:
        return None
    try:
        prices = {name: menu.model(name).price for name in pass_probs}
    except (KeyError, ValueError):
        return None

    expected_values = {
        name: (contract_price * _decimal(prob)) - prices[name]
        for name, prob in pass_probs.items()
    }
    oracle_best = max(expected_values, key=lambda name: expected_values[name])
    chosen_is_oracle = model == oracle_best
    passed = draw < pass_prob

    if passed:
        verdict = "passed"
        detail = (
            f"Passed: draw {draw:.2f} fell under the {pass_prob:.0%} pass rate for {model} "
            f"on a {difficulty} job."
        )
    elif chosen_is_oracle:
        verdict = "unlucky_rng"
        detail = (
            f"Unlucky RNG, not a bad pick. {model} was the highest expected-value choice "
            f"({pass_prob:.0%} pass on a {difficulty} job); it failed only because "
            f"draw {draw:.2f} exceeded {pass_prob:.2f}."
        )
    else:
        verdict = "suboptimal_model"
        detail = (
            f"Sub-optimal model. {model} ({pass_prob:.0%} pass, EV {_round(expected_values[model])}) "
            f"was beaten by {oracle_best} ({pass_probs[oracle_best]:.0%} pass, "
            f"EV {_round(expected_values[oracle_best])}) on this {difficulty} job."
        )

    return {
        "model": model,
        "internal_difficulty": difficulty,
        "task_type": task_type,
        "passed": passed,
        "pass_prob": pass_prob,
        "draw": round(draw, 4),
        "contract_price": str(contract_price),
        "oracle_best_model": oracle_best,
        "chosen_is_oracle_best": chosen_is_oracle,
        "verdict": verdict,
        "verdict_detail": detail,
        "models": [
            {
                "name": name,
                "pass_prob": pass_probs[name],
                "price": str(prices[name]),
                "expected_value": _round(expected_values[name]),
                "chosen": name == model,
                "oracle_best": name == oracle_best,
            }
            for name in pass_probs
        ],
    }


def _load_agent_turns(sidecar_path: Path) -> list[dict[str, Any]]:
    """Slim per-turn agent record (reasoning + tool call + result + tokens).

    Drops the heavy ``context_history`` the sidecar repeats every turn; keeps only
    what a human needs to follow the agent's decisions.
    """
    if not sidecar_path.exists():
        return []
    turns: list[dict[str, Any]] = []
    for line in sidecar_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # pragma: no cover - tolerate partial writes
            continue
        response = record.get("response", {}) or {}
        dispatch = record.get("dispatch", {}) or {}
        tool_call = response.get("tool_call") or {}
        request = record.get("request", {}) or {}
        turns.append(
            {
                "turn": record.get("turn"),
                # The model's genuine chain-of-thought, when the provider returns it.
                "reasoning": response.get("reasoning", "") or "",
                # The visible assistant prose emitted alongside the tool call.
                "message": response.get("content", "") or "",
                "tool": tool_call.get("name"),
                "arguments": tool_call.get("arguments", {}),
                "ok": dispatch.get("ok"),
                "result": dispatch.get("result"),
                "error": dispatch.get("error"),
                "usage": response.get("usage", {}),
                "request_tokens_estimate": request.get("request_tokens_estimate"),
            }
        )
    return turns


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
    if kind == "clarified":
        return f"{payload.get('job_id')} clarified"
    if kind == "models_listed":
        return f"{len(payload.get('models', []))} delivery model(s)"
    if kind == "delivered":
        return f"{payload.get('job_id')} via {payload.get('model')}"
    if kind == "tool_price_charged":
        return f"{payload.get('job_id')} charged {payload.get('price_charged')}"
    if kind in {"delivery_passed", "delivery_failed"}:
        return f"{payload.get('job_id')} {kind.replace('delivery_', '')} via {payload.get('model')}"
    if kind == "brain_metered":
        return f"{payload.get('model')} cost {payload.get('cost')}"
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


def _decimal(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _round(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01")))


def _relative_path(path: Path | None, root_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
