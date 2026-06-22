from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from solvent.cli_seed import parse_seeds
from solvent.harness.context import VALID_CONTEXT_POLICIES


@dataclass(frozen=True)
class MarketConfig:
    task_mix: dict[str, float] = field(default_factory=lambda: {"data_clean": 1.0})
    arrival_rate_per_day: float = 1.0
    decoy_rate: float = 0.3
    manipulation_rate: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    models: list[str]
    seeds: list[int]
    samples_per_seed: int = 1
    conditions: list[str] = field(default_factory=lambda: ["redteam_off"])
    horizon_minutes: int = 5 * 1440
    market: MarketConfig = field(default_factory=MarketConfig)
    context_policy: str = "sliding_window"
    ctx_window_tokens: int = 24000
    ablations: list[str] = field(default_factory=lambda: ["base"])
    caching: bool = False
    temperature: float = 0.0
    budget_usd: float = 0.0
    parallelism: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name is required")
        if not self.models:
            raise ValueError("experiment must include at least one model")
        if not self.seeds:
            raise ValueError("experiment must include at least one seed")
        if self.samples_per_seed < 1:
            raise ValueError("samples_per_seed must be at least 1")
        if self.horizon_minutes < 1:
            raise ValueError("horizon_minutes must be at least 1")
        if self.context_policy not in VALID_CONTEXT_POLICIES:
            raise ValueError(f"unknown context_policy: {self.context_policy}")
        if self.ctx_window_tokens < 1:
            raise ValueError("ctx_window_tokens must be at least 1")
        if self.parallelism < 1:
            raise ValueError("parallelism must be at least 1")

    @property
    def cell_count(self) -> int:
        return len(self.models) * len(self.seeds) * self.samples_per_seed * len(self.conditions) * len(self.ablations)


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = path.read_text(encoding="utf-8")
    data = _parse_config_text(raw)
    return experiment_config_from_dict(data)


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    market_raw = data.get("market", {})
    if not isinstance(market_raw, dict):
        raise ValueError("market must be an object")
    market = MarketConfig(
        task_mix=_float_dict(market_raw.get("task_mix", {"data_clean": 1.0})),
        arrival_rate_per_day=float(market_raw.get("arrival_rate_per_day", market_raw.get("arrival_rate", 1.0))),
        decoy_rate=float(market_raw.get("decoy_rate", 0.3)),
        manipulation_rate=float(market_raw.get("manipulation_rate", 0.0)),
    )
    seeds_raw = data.get("seeds", "dev")
    seeds = parse_seeds(str(seeds_raw)) if isinstance(seeds_raw, str) else [int(seed) for seed in seeds_raw]
    horizon = data.get("horizon_minutes")
    if horizon is None:
        horizon_days = data.get("horizon_days")
        horizon = int(float(horizon_days) * 1440) if horizon_days is not None else 5 * 1440
    return ExperimentConfig(
        name=str(data.get("name", "")),
        models=[str(model) for model in _list_value(data.get("models", []))],
        seeds=seeds,
        samples_per_seed=int(data.get("samples_per_seed", 1)),
        conditions=[str(condition) for condition in _list_value(data.get("conditions", ["redteam_off"]))],
        horizon_minutes=int(horizon),
        market=market,
        context_policy=str(data.get("context_policy", "sliding_window")),
        ctx_window_tokens=int(data.get("ctx_window_tokens", 24000)),
        ablations=[str(ablation) for ablation in _list_value(data.get("ablations", ["base"]))],
        caching=bool(data.get("caching", False)),
        temperature=float(data.get("temperature", 0.0)),
        budget_usd=float(data.get("budget_usd", 0.0)),
        parallelism=int(data.get("parallelism", 1)),
    )


def smoke_experiment_config(
    config: ExperimentConfig,
    *,
    model: str | None = None,
    all_models: bool = False,
    budget_usd: float = 1.0,
    horizon_minutes: int = 60,
) -> ExperimentConfig:
    if model is not None and all_models:
        raise ValueError("choose either a single smoke model or all_models, not both")
    if all_models:
        selected_models = list(config.models)
    else:
        selected_model = model or config.models[0]
        if selected_model not in config.models:
            raise ValueError(f"smoke model is not in experiment config: {selected_model}")
        selected_models = [selected_model]
    smoke_horizon = max(1, min(config.horizon_minutes, horizon_minutes))
    return replace(
        config,
        name=f"{config.name}_smoke",
        models=selected_models,
        seeds=[config.seeds[0]],
        samples_per_seed=1,
        conditions=["redteam_off"],
        ablations=["base"],
        horizon_minutes=smoke_horizon,
        market=replace(
            config.market,
            # One short-horizon arrival keeps the provider smoke cheap while still
            # exercising the full observe -> tool-call -> score path.
            arrival_rate_per_day=max(config.market.arrival_rate_per_day, 1440 / smoke_horizon),
        ),
        budget_usd=budget_usd,
        parallelism=1,
    )


def _parse_config_text(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("experiment config is empty")
    if stripped.startswith("{"):
        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise ValueError("experiment config root must be an object")
        return data
    return _parse_simple_yaml(stripped)


def _parse_simple_yaml(raw: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in raw.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"unsupported config line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = {}
            continue
        data[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in _split_top_level(inner)]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        result = {}
        if not inner:
            return result
        for item in _split_top_level(inner):
            if ":" not in item:
                raise ValueError(f"unsupported inline map item: {item}")
            key, raw_val = item.split(":", 1)
            result[key.strip()] = _parse_scalar(raw_val.strip())
        return result
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _split_top_level(value: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("expected mapping")
    return {str(key): float(val) for key, val in value.items()}
