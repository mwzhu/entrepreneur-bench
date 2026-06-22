from __future__ import annotations

from pathlib import Path


def parse_seeds(raw: str) -> list[int]:
    if raw in {"dev", "test"}:
        return read_seed_split(raw)
    path = Path(raw)
    if path.exists():
        return read_seed_file(path)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def seed_split_label(raw: str) -> str:
    return raw if raw in {"dev", "test"} else "ad_hoc"


def read_seed_split(name: str) -> list[int]:
    return read_seed_file(Path(__file__).resolve().parent / "configs" / f"seeds_{name}.txt")


def read_seed_file(path: Path) -> list[int]:
    seeds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            seeds.append(int(line))
    return seeds
