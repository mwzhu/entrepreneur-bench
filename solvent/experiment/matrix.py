from __future__ import annotations

import hashlib
from dataclasses import dataclass

from solvent.experiment.config import ExperimentConfig

VALID_CONDITIONS = {"redteam_off", "redteam_on"}


@dataclass(frozen=True)
class ExperimentCell:
    cell_id: str
    model: str
    config_id: str
    ablation: str
    seed: int
    sample_index: int
    condition: str
    redteam_enabled: bool

    @property
    def model_family(self) -> str:
        return self.model.split(":", 1)[0]


def expand_matrix(config: ExperimentConfig) -> list[ExperimentCell]:
    unknown_conditions = set(config.conditions) - VALID_CONDITIONS
    if unknown_conditions:
        raise ValueError(f"unknown experiment conditions: {', '.join(sorted(unknown_conditions))}")

    cells: list[ExperimentCell] = []
    for model in config.models:
        for ablation in config.ablations:
            config_id = compose_config_id(model, ablation)
            for seed in config.seeds:
                for sample_index in range(config.samples_per_seed):
                    for condition in config.conditions:
                        redteam_enabled = condition == "redteam_on"
                        cells.append(
                            ExperimentCell(
                                cell_id=_cell_id(model, ablation, seed, sample_index, condition),
                                model=model,
                                config_id=config_id,
                                ablation=ablation,
                                seed=seed,
                                sample_index=sample_index,
                                condition=condition,
                                redteam_enabled=redteam_enabled,
                            )
                        )
    return cells


def compose_config_id(model: str, ablation: str) -> str:
    family, spec = model.split(":", 1) if ":" in model else (model, "base")
    if ablation == "base":
        return f"{family}:{spec or 'base'}"
    if ablation.startswith("+"):
        if spec in {"", "base"}:
            return f"{family}:{ablation}"
        return f"{family}:{spec}{ablation}"
    return f"{family}:{ablation}"


def _cell_id(model: str, ablation: str, seed: int, sample_index: int, condition: str) -> str:
    raw = f"{model}|{ablation}|{seed}|{sample_index}|{condition}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    safe_model = model.replace(":", "-").replace("/", "-")
    safe_ablation = ablation.replace("+", "plus-").replace(":", "-").replace("/", "-")
    return f"{safe_model}-{safe_ablation}-seed-{seed}-sample-{sample_index}-{condition}-{digest}"
