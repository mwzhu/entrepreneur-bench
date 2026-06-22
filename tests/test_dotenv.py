import os
from pathlib import Path

from solvent.dotenv import load_dotenv


def test_load_dotenv_reads_local_credentials_without_overriding(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# local credentials",
                "OPENAI_API_KEY=from-file",
                "GOOGLE_API_KEY='quoted-file-value'",
                "EXISTING_KEY=file-should-not-win",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING_KEY", "from-shell")

    loaded = load_dotenv(env_path)

    assert loaded == 2
    assert os.environ["OPENAI_API_KEY"] == "from-file"
    assert os.environ["GOOGLE_API_KEY"] == "quoted-file-value"
    assert os.environ["EXISTING_KEY"] == "from-shell"


def test_load_dotenv_can_be_disabled(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SOLVENT_DOTENV", "0")

    assert load_dotenv(env_path) == 0
    assert "OPENAI_API_KEY" not in os.environ
