from solvent.tasks.data_clean import generate_data_clean_job
from solvent.tasks.extract import generate_extract_job


def test_data_clean_difficulty_knob_is_reproducible_and_structural() -> None:
    easy = generate_data_clean_job(42, 0, "easy")
    easy_again = generate_data_clean_job(42, 0, "easy")
    med = generate_data_clean_job(42, 0, "med")
    hard = generate_data_clean_job(42, 0, "hard")

    assert easy.inputs == easy_again.inputs
    assert easy.internal_difficulty == "easy"
    assert med.internal_difficulty == "med"
    assert hard.internal_difficulty == "hard"
    assert _row_count(easy.inputs["csv"]) < _row_count(med.inputs["csv"]) < _row_count(hard.inputs["csv"])


def test_extract_difficulty_knob_is_reproducible_and_adds_distractors() -> None:
    easy = generate_extract_job(42, 0, "easy")
    easy_again = generate_extract_job(42, 0, "easy")
    med = generate_extract_job(42, 0, "med")
    hard = generate_extract_job(42, 0, "hard")

    assert easy.inputs == easy_again.inputs
    assert "Previous invoice" not in easy.inputs["document"]
    assert "Previous invoice" not in med.inputs["document"]
    assert "Previous invoice" in hard.inputs["document"]
    assert len(easy.inputs["document"]) < len(med.inputs["document"]) < len(hard.inputs["document"])


def _row_count(csv_text: str) -> int:
    return max(0, len(csv_text.strip().splitlines()) - 1)
