from pathlib import Path

import pytest

from solvent.demo import CompareOptions, run_compare_artifact


class CrashingHarness:
    def run(self, env) -> None:
        raise RuntimeError("harness exploded")


def test_compare_artifact_surfaces_harness_exceptions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("solvent.demo.harness_from_config_id", lambda config_id, **kwargs: CrashingHarness())
    with pytest.raises(RuntimeError, match="harness exploded"):
        run_compare_artifact(
            CompareOptions(
                config_a="stub:naive",
                config_b="stub:procedure",
                seeds=[42],
                trace_dir=tmp_path / "compare",
            )
        )


def test_compare_options_rejects_non_positive_samples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="samples"):
        CompareOptions(
            config_a="stub:naive",
            config_b="stub:procedure",
            seeds=[42],
            trace_dir=tmp_path / "compare",
            samples=0,
        )


def test_compare_options_rejects_out_of_range_temperature(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="temperature"):
        CompareOptions(
            config_a="stub:naive",
            config_b="stub:procedure",
            seeds=[42],
            trace_dir=tmp_path / "compare",
            temperature=1.1,
        )


def test_compare_options_rejects_non_positive_model_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_max_turns"):
        CompareOptions(
            config_a="stub:naive",
            config_b="stub:procedure",
            seeds=[42],
            trace_dir=tmp_path / "compare-turns",
            model_max_turns=0,
        )
    with pytest.raises(ValueError, match="model_max_tokens"):
        CompareOptions(
            config_a="stub:naive",
            config_b="stub:procedure",
            seeds=[42],
            trace_dir=tmp_path / "compare-tokens",
            model_max_tokens=0,
        )
