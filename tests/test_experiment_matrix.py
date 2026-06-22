from solvent.experiment.config import experiment_config_from_dict
from solvent.experiment.matrix import compose_config_id, expand_matrix


def test_expand_matrix_is_deterministic_and_covers_all_dimensions() -> None:
    config = experiment_config_from_dict(
        {
            "name": "matrix",
            "models": ["fake:base", "claude-sonnet-4-6:base"],
            "seeds": [1, 2],
            "samples_per_seed": 2,
            "conditions": ["redteam_off", "redteam_on"],
            "ablations": ["base", "+procedure"],
        }
    )

    first = expand_matrix(config)
    second = expand_matrix(config)

    assert len(first) == config.cell_count == 32
    assert [cell.cell_id for cell in first] == [cell.cell_id for cell in second]
    assert len({cell.cell_id for cell in first}) == len(first)
    assert {cell.redteam_enabled for cell in first} == {False, True}


def test_compose_config_id_keeps_base_and_appends_plus_ablations() -> None:
    assert compose_config_id("claude-opus-4-8:base", "base") == "claude-opus-4-8:base"
    assert compose_config_id("claude-opus-4-8:base", "+procedure") == "claude-opus-4-8:+procedure"
    assert compose_config_id("claude-opus-4-8:planner", "+procedure") == "claude-opus-4-8:planner+procedure"
