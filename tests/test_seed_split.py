from pathlib import Path

from solvent.cli_seed import parse_seeds, read_seed_file, read_seed_split, seed_split_label


def test_named_seed_splits_resolve_to_frozen_lists() -> None:
    assert read_seed_split("dev") == [40, 41, 42, 43, 44]
    assert read_seed_split("test") == [140, 141, 142, 143, 144]
    assert parse_seeds("dev") == [40, 41, 42, 43, 44]
    assert parse_seeds("test") == [140, 141, 142, 143, 144]


def test_seed_split_label_marks_only_named_splits() -> None:
    assert seed_split_label("dev") == "dev"
    assert seed_split_label("test") == "test"
    assert seed_split_label("40,41") == "ad_hoc"


def test_parse_seeds_accepts_ad_hoc_lists_and_seed_files(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text(
        """
        # tuning sample
        7
        8 # inline comments are ignored

        9
        """,
        encoding="utf-8",
    )

    assert parse_seeds("1, 2,3") == [1, 2, 3]
    assert read_seed_file(seed_file) == [7, 8, 9]
    assert parse_seeds(str(seed_file)) == [7, 8, 9]
