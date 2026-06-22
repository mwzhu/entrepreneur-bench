from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solvent.characterize import validate_menu
from solvent.cli_seed import read_seed_split
from solvent.env.pricing import price_for_model
from solvent.experiment.config import load_experiment_config
from solvent.experiment.estimate import estimate_experiment
from solvent.harness.model_client import model_alias_env_var, resolve_model_name
from solvent.harness.providers.openai_compat import OPENROUTER_API_KEY_ENV, probe_openrouter_chat, provider_for_model


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def doctor(agent: str = "claude-opus-4-8:base", *, probe_live: bool = False) -> dict[str, Any]:
    checks = [
        _menu_check(),
        _seed_split_check("dev"),
        _seed_split_check("test"),
        _replay_check(),
        _pricing_check(agent),
        _live_model_check(agent, probe_live=probe_live),
    ]
    return {
        "ok": all(check.ok for check in checks),
        "agent": agent,
        "checks": [check.__dict__ for check in checks],
    }


def experiment_doctor(config_path: Path, *, probe_live: bool = False) -> dict[str, Any]:
    checks = [
        _menu_check(),
        _seed_split_check("dev"),
        _seed_split_check("test"),
        _replay_check(),
    ]
    try:
        config = load_experiment_config(config_path)
        estimate = estimate_experiment(config)
    except (OSError, ValueError, KeyError) as exc:
        checks.append(CheckResult(name="experiment_config", ok=False, detail=str(exc)))
        return {
            "ok": False,
            "config_path": str(config_path),
            "checks": [check.__dict__ for check in checks],
        }

    checks.append(
        CheckResult(
            name="experiment_config",
            ok=True,
            detail=f"{config.name}; models={len(config.models)} seeds={len(config.seeds)} cells={config.cell_count}",
        )
    )
    checks.append(
        CheckResult(
            name="experiment_estimate",
            ok=not estimate.over_budget,
            detail=f"total_cost={estimate.total_cost} budget_usd={estimate.budget_usd} over_budget={estimate.over_budget}",
        )
    )
    checks.append(
        CheckResult(
            name="experiment_caching",
            ok=True,
            detail=(
                "caching requested; verify non-zero cache-read counters after the first live run"
                if config.caching
                else "caching disabled in experiment config"
            ),
        )
    )
    for model in config.models:
        checks.append(_pricing_check(model))
        checks.append(_live_model_check(model, probe_live=probe_live))
    return {
        "ok": all(check.ok for check in checks),
        "config_path": str(config_path),
        "name": config.name,
        "estimate": estimate.to_dict(),
        "checks": [check.__dict__ for check in checks],
    }


def _menu_check() -> CheckResult:
    validation = validate_menu()
    return CheckResult(
        name="delivery_menu",
        ok=validation.valid,
        detail=f"{validation.version} {validation.checksum}",
    )


def _seed_split_check(name: str) -> CheckResult:
    try:
        seeds = read_seed_split(name)
    except (OSError, ValueError) as exc:
        return CheckResult(name=f"seed_split_{name}", ok=False, detail=str(exc))
    return CheckResult(name=f"seed_split_{name}", ok=bool(seeds), detail=",".join(str(seed) for seed in seeds))


def _replay_check() -> CheckResult:
    return CheckResult(name="recorded_replay", ok=True, detail="score/replay paths do not require model credentials")


def _pricing_check(agent: str) -> CheckResult:
    family = agent.split(":", 1)[0]
    if family in {"stub", "fake", "recorded"}:
        return CheckResult(name="brain_pricing", ok=True, detail=f"{family} is local/no-cost")
    try:
        price = price_for_model(family)
    except KeyError as exc:
        return CheckResult(name="brain_pricing", ok=False, detail=str(exc))
    return CheckResult(name="brain_pricing", ok=True, detail=f"{price.version}; verified {price.verified_date}; {price.source_url}")


def _live_model_check(agent: str, *, probe_live: bool = False) -> CheckResult:
    family = agent.split(":", 1)[0]
    if family == "stub":
        return CheckResult(name="live_model_credentials", ok=True, detail="stub config does not require live credentials")
    if family == "fake":
        return CheckResult(name="live_model_credentials", ok=True, detail="fake client is local-only")
    if family.startswith("claude-"):
        ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
        alias_var = model_alias_env_var(family)
        resolved = resolve_model_name(family)
        alias_detail = f"; {alias_var}={resolved}" if resolved != family else f"; {alias_var} not set"
        return CheckResult(
            name="live_model_credentials",
            ok=ok,
            detail=("ANTHROPIC_API_KEY present" if ok else "ANTHROPIC_API_KEY missing") + alias_detail,
        )
    if family.startswith("gemini-"):
        ok = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        return CheckResult(
            name="live_model_credentials",
            ok=ok,
            detail="GOOGLE_API_KEY/GEMINI_API_KEY present" if ok else "GOOGLE_API_KEY or GEMINI_API_KEY missing",
        )
    provider = provider_for_model(family)
    if provider is not None:
        native_ok = bool(os.environ.get(provider.api_key_env))
        openrouter_ok = bool(provider.openrouter_model and os.environ.get(OPENROUTER_API_KEY_ENV))
        ok = native_ok or openrouter_ok
        if native_ok:
            detail = f"{provider.api_key_env} present"
        elif openrouter_ok:
            alias_var = model_alias_env_var(family)
            resolved = resolve_model_name(family)
            model_detail = resolved if resolved != family else provider.openrouter_model
            detail = f"{OPENROUTER_API_KEY_ENV} present via OpenRouter model {model_detail}; {alias_var}"
            if probe_live:
                probe_ok, probe_detail = probe_openrouter_chat(family)
                ok = probe_ok
                detail = probe_detail
        elif provider.openrouter_model:
            detail = f"{provider.api_key_env} or {OPENROUTER_API_KEY_ENV} missing"
        else:
            detail = f"{provider.api_key_env} missing"
        return CheckResult(
            name="live_model_credentials",
            ok=ok,
            detail=detail,
        )
    return CheckResult(name="live_model_credentials", ok=False, detail=f"no live client configured for {family}")
