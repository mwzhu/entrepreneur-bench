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
    assert all("cell_id" in run for run in manifest["runs"])
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


def test_build_viewer_preserves_multiple_samples_in_manifest_and_ui(tmp_path: Path) -> None:
    artifact = run_compare_artifact(
        CompareOptions(
            config_a="stub:naive",
            config_b="stub:procedure",
            seeds=[42],
            samples=2,
            trace_dir=tmp_path / "samples",
        )
    )

    build_viewer(artifact.trace_dir, artifact.summary, artifact.runs)

    manifest = json.loads((artifact.trace_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["samples"] == [0, 1]
    assert len(manifest["runs"]) == 4
    assert {
        (run["config_id"], run["seed"], run["sample_index"], run["redteam_enabled"])
        for run in manifest["runs"]
    } == {
        ("stub:naive", 42, 0, False),
        ("stub:naive", 42, 1, False),
        ("stub:procedure", 42, 0, False),
        ("stub:procedure", 42, 1, False),
    }

    index_html = (artifact.trace_dir / "viewer" / "index.html").read_text(encoding="utf-8")
    app_js = (artifact.trace_dir / "viewer" / "app.js").read_text(encoding="utf-8")
    assert 'id="sample-select"' in index_html
    assert "String(item.sample_index || 0) === sampleSelect.value" in app_js


def _load_data_js(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    prefix = "window.SOLVENT_DATA = "
    assert text.startswith(prefix)
    assert text.endswith(";\n")
    return json.loads(text[len(prefix) : -2])
