from solvent.scoring.aggregate import summarize_distribution


def test_summarize_distribution_includes_95_percent_ci_for_multiple_samples() -> None:
    summary = summarize_distribution([1, 2, 3, None])

    assert summary.mean == 2.0
    assert round(summary.std or 0, 6) == 1.0
    assert summary.min == 1.0
    assert summary.n == 3
    assert round(summary.ci95_low or 0, 6) == 0.868393
    assert round(summary.ci95_high or 0, 6) == 3.131607
    assert summary.to_dict()["ci95_low"] == summary.ci95_low


def test_summarize_distribution_leaves_ci_empty_for_single_sample() -> None:
    summary = summarize_distribution([4])

    assert summary.mean == 4.0
    assert summary.std == 0.0
    assert summary.ci95_low is None
    assert summary.ci95_high is None


def test_summarize_distribution_empty_values_have_empty_ci() -> None:
    summary = summarize_distribution([None])

    assert summary.to_dict() == {
        "mean": None,
        "std": None,
        "min": None,
        "n": 0,
        "ci95_low": None,
        "ci95_high": None,
    }
