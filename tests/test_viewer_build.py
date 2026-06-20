import json
from pathlib import Path

from solvent.demo import CompareOptions, run_compare_artifact
from solvent.viewer.build import build_viewer


def test_build_viewer_writes_manifest_json_views_and_data_js(tmp_path: Path) -> None:
    artifact = run_compare_artifact(
        CompareOptions(
            config_a="stub:naive",
            config_b="stub:procedure",
            seeds=[42],
            trace_dir=tmp_path / "demo",
            redteam_paired=True,
        )
    )

    viewer_path = build_viewer(artifact.trace_dir, artifact.summary, artifact.runs)

    assert viewer_path.exists()
    assert (artifact.trace_dir / "viewer" / "app.js").exists()
    assert (artifact.trace_dir / "viewer" / "style.css").exists()
    assert (artifact.trace_dir / "manifest.json").exists()
    assert (artifact.trace_dir / "viewer" / "data.js").exists()

    manifest = json.loads((artifact.trace_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["runs"]) == 4
    assert all(not Path(run["trace_path"]).is_absolute() for run in manifest["runs"])
    assert all((artifact.trace_dir / run["view_path"]).exists() for run in manifest["runs"])

    data_payload = _load_data_js(artifact.trace_dir / "viewer" / "data.js")
    assert data_payload["manifest"] == manifest
    assert data_payload["summary"] == artifact.summary
    assert set(data_payload["traces"]) == {run["key"] for run in manifest["runs"]}

    before = (artifact.trace_dir / "viewer" / "data.js").read_text(encoding="utf-8")
    build_viewer(artifact.trace_dir, artifact.summary, artifact.runs)
    after = (artifact.trace_dir / "viewer" / "data.js").read_text(encoding="utf-8")
    assert before == after


def _load_data_js(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    prefix = "window.SOLVENT_DATA = "
    assert text.startswith(prefix)
    assert text.endswith(";\n")
    return json.loads(text[len(prefix) : -2])
