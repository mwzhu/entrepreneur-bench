from pathlib import Path

import solvent


def test_package_version_tracks_v0_5_milestone() -> None:
    project_version = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in Path("pyproject.toml").read_text(encoding="utf-8").splitlines()
        if line.startswith("version = ")
    )

    assert project_version == "0.5.0"
    assert solvent.__version__ == "0.5.0"
