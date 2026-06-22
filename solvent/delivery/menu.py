from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from decimal import Decimal
from importlib import resources
from typing import Any


DEFAULT_MENU_RESOURCE = "menu_v0_4.json"
DIFFICULTIES = ("easy", "med", "hard")


@dataclass(frozen=True)
class ModelPublic:
    name: str
    price: Decimal
    capability_proxy: float
    speed_proxy: str


@dataclass(frozen=True)
class DeliveryResolution:
    passed: bool
    price_charged: Decimal
    duration: int
    pass_prob: float
    draw: float = 0.0


def delivery_draw_key(seed: int, job_id: str, model: str, attempt_index: int, menu_version: str) -> str:
    """Stable RNG key for a delivery draw.

    Shared by the environment (which performs the draw) and the viewer (which
    reconstructs it for post-hoc diagnosis), so the two never drift.
    """
    return f"{seed}:{job_id}:{model}:{attempt_index}:{menu_version}"


class DeliveryMenu:
    def __init__(self, data: dict[str, Any], checksum: str):
        self.data = data
        self.checksum = checksum
        self.schema_version = str(data["schema_version"])
        self.version = str(data.get("version", "menu_v0_4"))
        self.source = str(data.get("source", "unknown"))
        self.calibration = data.get("calibration", {})
        self._tools = {
            str(tool["name"]): ModelPublic(
                name=str(tool["name"]),
                price=Decimal(str(tool["price"])),
                capability_proxy=float(tool["capability_proxy"]),
                speed_proxy=str(tool["speed_proxy"]),
            )
            for tool in data["tools"]
        }
        self._profile = data["profile"]
        self._validate()

    @classmethod
    def load_default(cls) -> "DeliveryMenu":
        raw = resources.files("solvent.delivery.menu_data").joinpath(DEFAULT_MENU_RESOURCE).read_text(encoding="utf-8")
        return cls.from_json(raw)

    @classmethod
    def from_json(cls, raw: str) -> "DeliveryMenu":
        data = json.loads(raw)
        checksum = hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()
        return cls(data, checksum)

    def public_models(self) -> list[ModelPublic]:
        return list(self._tools.values())

    def model(self, name: str) -> ModelPublic:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(f"unknown delivery model: {name}") from exc

    def pass_prob(self, task_type: str, model: str, difficulty: str) -> float:
        return float(self._profile[task_type][model][difficulty]["pass"])

    def pass_prob_by_model(self, task_type: str, difficulty: str) -> dict[str, float]:
        """Pass probability of every model for one (task, difficulty) cell."""
        return {name: self.pass_prob(task_type, name, difficulty) for name in self._tools}

    def duration(self, task_type: str, model: str, difficulty: str) -> int:
        return int(self._profile[task_type][model][difficulty]["duration"])

    def resolve(self, task_type: str, model: str, difficulty: str, draw_key: str) -> DeliveryResolution:
        public = self.model(model)
        pass_prob = self.pass_prob(task_type, model, difficulty)
        rng = random.Random(draw_key)
        draw = rng.random()
        return DeliveryResolution(
            passed=draw < pass_prob,
            price_charged=public.price,
            duration=self.duration(task_type, model, difficulty),
            pass_prob=pass_prob,
            draw=draw,
        )

    def _validate(self) -> None:
        if self.schema_version != "solvent_delivery_menu_v0_4":
            raise ValueError(f"unsupported delivery menu schema: {self.schema_version}")
        if self.source not in {"calibrated_synthetic", "characterized"}:
            raise ValueError(f"unsupported delivery menu source: {self.source}")
        if not isinstance(self.calibration, dict) or not self.calibration.get("basis"):
            raise ValueError("delivery menu missing calibration provenance")
        if not self._tools:
            raise ValueError("delivery menu has no tools")
        for model_name, model in self._tools.items():
            if model.price <= 0:
                raise ValueError(f"delivery model price must be positive: {model_name}")
            if not 0 <= model.capability_proxy <= 1:
                raise ValueError(f"capability proxy out of range: {model_name}")
        for task_type, task_profile in self._profile.items():
            missing = set(self._tools) - set(task_profile)
            if missing:
                raise ValueError(f"task {task_type} missing tool profiles: {sorted(missing)}")
            extra = set(task_profile) - set(self._tools)
            if extra:
                raise ValueError(f"task {task_type} has unknown tool profiles: {sorted(extra)}")
            for model_name, difficulty_profile in task_profile.items():
                previous_pass = None
                previous_duration = None
                for difficulty in DIFFICULTIES:
                    if difficulty not in difficulty_profile:
                        raise ValueError(f"{task_type}/{model_name} missing difficulty: {difficulty}")
                    cell = difficulty_profile[difficulty]
                    pass_prob = float(cell["pass"])
                    duration = int(cell["duration"])
                    if not 0 <= pass_prob <= 1:
                        raise ValueError(f"pass probability out of range: {task_type}/{model_name}/{difficulty}")
                    if duration <= 0:
                        raise ValueError(f"duration must be positive: {task_type}/{model_name}/{difficulty}")
                    if previous_pass is not None and pass_prob > previous_pass:
                        raise ValueError(f"pass probability not monotone: {task_type}/{model_name}")
                    if previous_duration is not None and duration < previous_duration:
                        raise ValueError(f"duration not monotone: {task_type}/{model_name}")
                    previous_pass = pass_prob
                    previous_duration = duration
        dominated = self._dominated_tools()
        if dominated:
            raise ValueError(f"delivery menu has dominated tools: {sorted(dominated)}")

    def _dominated_tools(self) -> set[str]:
        dominated: set[str] = set()
        for target in self._tools.values():
            for challenger in self._tools.values():
                if challenger.name == target.name:
                    continue
                if self._dominates(challenger.name, target.name):
                    dominated.add(target.name)
                    break
        return dominated

    def _dominates(self, challenger_name: str, target_name: str) -> bool:
        challenger = self._tools[challenger_name]
        target = self._tools[target_name]
        if challenger.price > target.price:
            return False
        strictly_better = challenger.price < target.price
        for task_type in self._profile:
            for difficulty in DIFFICULTIES:
                challenger_pass = self.pass_prob(task_type, challenger_name, difficulty)
                target_pass = self.pass_prob(task_type, target_name, difficulty)
                challenger_duration = self.duration(task_type, challenger_name, difficulty)
                target_duration = self.duration(task_type, target_name, difficulty)
                if challenger_pass < target_pass or challenger_duration > target_duration:
                    return False
                strictly_better = (
                    strictly_better
                    or challenger_pass > target_pass
                    or challenger_duration < target_duration
                )
        return strictly_better


def _canonical(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))
