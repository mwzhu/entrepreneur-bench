from pathlib import Path

import pytest

from solvent.demo import CompareOptions, run_compare_artifact


class CrashingHarness:
    def run(self, env) -> None:
        raise RuntimeError("harness exploded")


def test_compare_artifact_surfaces_harness_exceptions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("solvent.demo.harness_from_config_id", lambda config_id: CrashingHarness())
    with pytest.raises(RuntimeError, match="harness exploded"):
        run_compare_artifact(
            CompareOptions(
                config_a="stub:naive",
                config_b="stub:procedure",
                seeds=[42],
                trace_dir=tmp_path / "compare",
            )
        )
