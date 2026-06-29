from __future__ import annotations

from datasets import Dataset

from solvent.cli_seed import parse_seeds, seed_split_label


def build_seed_dataset(split: str = "train", horizon_days: int = 2, config_id: str = "rl:qwen3-4b") -> Dataset:
    rows = []
    for seed in parse_seeds(split):
        rows.append(
            {
                "question": f"Run Solvent episode seed={seed}, horizon_days={horizon_days}.",
                "answer": "",
                "info": {
                    "seed": seed,
                    "config_id": config_id,
                    "split": seed_split_label(split),
                    "horizon_days": horizon_days,
                },
            }
        )
    return Dataset.from_list(rows)
