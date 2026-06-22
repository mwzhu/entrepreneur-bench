from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

from solvent import __version__
from solvent.demo import RunArtifact
from solvent.viewer.trace_view import build_trace_view


def build_viewer(trace_dir: Path, summary: dict[str, Any], runs: list[RunArtifact]) -> Path:
    trace_dir = trace_dir.resolve()
    viewer_dir = trace_dir / "viewer"
    data_dir = viewer_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _copy_static_files(viewer_dir)

    manifest = _manifest(trace_dir, summary, runs)
    views = {}
    for run in runs:
        view = build_trace_view(run.trace_path, run.scorecard_path, root_dir=trace_dir)
        view_name = _view_filename(run)
        view_path = data_dir / view_name
        view_path.write_text(_json(view) + "\n", encoding="utf-8")
        views[_run_key(run)] = view

    (trace_dir / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    (data_dir / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    (data_dir / "summary.json").write_text(_json(summary) + "\n", encoding="utf-8")

    data_payload = {"manifest": manifest, "summary": summary, "traces": views}
    (viewer_dir / "data.js").write_text("window.SOLVENT_DATA = " + _json(data_payload) + ";\n", encoding="utf-8")
    return viewer_dir / "index.html"


def _manifest(trace_dir: Path, summary: dict[str, Any], runs: list[RunArtifact]) -> dict[str, Any]:
    configs = []
    seeds = []
    samples = []
    for run in runs:
        if run.config_id not in configs:
            configs.append(run.config_id)
        if run.seed not in seeds:
            seeds.append(run.seed)
        if run.sample_index not in samples:
            samples.append(run.sample_index)
    return {
        "schema_version": "solvent_demo_v0_4",
        "created_by": f"solvent {__version__}",
        "summary_path": "summary.json",
        "viewer_entry": "viewer/index.html",
        "configs": configs,
        "seeds": sorted(seeds),
        "samples": sorted(samples),
        "redteam_paired": any(run.redteam_enabled for run in runs),
        "runs": [
            {
                "key": _run_key(run),
                "config_id": run.config_id,
                "cell_id": run.cell_id,
                "seed": run.seed,
                "sample_index": run.sample_index,
                "redteam_enabled": run.redteam_enabled,
                "trace_path": _relative_path(run.trace_path, trace_dir),
                "scorecard_path": _relative_path(run.scorecard_path, trace_dir),
                "view_path": f"viewer/data/{_view_filename(run)}",
            }
            for run in runs
        ],
        "metric_labels": summary.get("metric_labels", {}),
    }


def _copy_static_files(viewer_dir: Path) -> None:
    static_root = resources.files("solvent.viewer").joinpath("static")
    for name in ["index.html", "app.js", "style.css"]:
        source = static_root.joinpath(name)
        with resources.as_file(source) as source_path:
            shutil.copyfile(source_path, viewer_dir / name)


def _view_filename(run: RunArtifact) -> str:
    return f"{_run_key(run)}.view.json"


def _run_key(run: RunArtifact) -> str:
    config = run.config_id.replace(":", "-")
    redteam = "redteam-on" if run.redteam_enabled else "redteam-off"
    sample = f"-sample-{run.sample_index}" if run.sample_index else ""
    return f"seed-{run.seed}{sample}-{config}-{redteam}"


def _relative_path(path: Path, root_dir: Path) -> str:
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
