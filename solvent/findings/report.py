from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent.findings.leaderboard import FindingsData, build_findings_data
from solvent.viewer.build import build_viewer


def generate_findings(experiment_dir: Path) -> dict[str, Any]:
    data = build_findings_data(experiment_dir)
    summary_path = data.experiment_dir / "summary.json"
    leaderboard_path = data.experiment_dir / "leaderboard.json"
    report_path = data.experiment_dir / "findings.md"
    summary_path.write_text(_json(data.summary) + "\n", encoding="utf-8")
    leaderboard_path.write_text(_json({"leaderboard": data.leaderboard}) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(data), encoding="utf-8")
    viewer_path = build_viewer(data.experiment_dir, data.summary, data.runs)
    return {
        "summary_path": str(summary_path),
        "leaderboard_path": str(leaderboard_path),
        "findings_path": str(report_path),
        "viewer_path": str(viewer_path),
        "leaderboard": data.leaderboard,
    }


def _markdown_report(data: FindingsData) -> str:
    lines = [
        f"# Solvent Findings: {data.summary['name']}",
        "",
        "## Leaderboard",
        "",
        "| rank | config | net mean | net std | net min | net 95% CI | fraction optimal | manipulation loss | jobs delivered | days until insolvent | horizon active | compute cost | cache hit | efficiency | completed | censored |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if any(row["omniscient_reference_relaxation"] or row["realizable_reference_relaxation"] for row in data.leaderboard):
        lines.extend(
            [
                "> Reference caveat: one or more cells used the large-stream upper-bound relaxation rather than the exact expiry-aware DP. Fraction-of-optimal values for those rows are lower-bound fractions against an upper-bound reference, not exact optimum fractions.",
                "",
            ]
        )
    for row in data.leaderboard:
        lines.append(
            "| {rank} | {config} | {net_mean} | {net_std} | {net_min} | {net_ci} | {frac} | {manipulation} | {jobs} | {days} | {horizon} | {compute} | {cache_hit} | {efficiency} | {completed} | {censored} |".format(
                rank=row["rank"],
                config=row["config_id"],
                net_mean=_fmt(row["net_revenue"]["mean"]),
                net_std=_fmt(row["net_revenue"]["std"]),
                net_min=_fmt(row["net_revenue"]["min"]),
                net_ci=_fmt_ci(row["net_revenue"]),
                frac=_fmt(row["fraction_of_omniscient_optimal"]["mean"]),
                manipulation=_fmt(row["manipulation_resistance_loss"]["mean"]),
                jobs=_fmt(row["jobs_delivered"]["mean"]),
                days=_fmt(row["days_until_insolvent"]["mean"]),
                horizon=_fmt(row["horizon_fraction_active"]["mean"]),
                compute=_fmt(row["brain_compute_cost"]["mean"]),
                cache_hit=_fmt(row["brain_cache_hit_rate"]["mean"]),
                efficiency=_fmt(row["efficiency"]["mean"]),
                completed=row["completed_cells"],
                censored=row["censored_cells"],
            )
        )
    lines.extend(["", "## Capability Decomposition", ""])
    lines.append("| config | selection regret | pricing regret | delivery pass | tool regret | support conceded | manipulation loss | coherence penalty |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in data.leaderboard:
        lines.append(
            "| {config} | {selection} | {pricing} | {delivery} | {tool} | {support} | {manipulation} | {coherence} |".format(
                config=row["config_id"],
                selection=_fmt(row["selection_regret"]["mean"]),
                pricing=_fmt(row["pricing_regret"]["mean"]),
                delivery=_fmt(row["delivery_pass_rate"]["mean"]),
                tool=_fmt(row["tool_selection_regret"]["mean"]),
                support=_fmt(row["support_conceded_value"]["mean"]),
                manipulation=_fmt(row["manipulation_resistance_loss"]["mean"]),
                coherence=_fmt(row["coherence_penalty"]["mean"]),
            )
        )
    lines.extend(["", "## Model Notes", ""])
    for row in data.leaderboard:
        lines.append(_model_note(row))
    lines.extend(["", "## Cache Verification", ""])
    lines.append("| config | status | cache-read tokens | cache-write tokens | detail |")
    lines.append("|---|---|---:|---:|---|")
    for row in data.leaderboard:
        cache = row["cache_verification"]
        lines.append(
            "| {config} | {status} | {read} | {write} | {detail} |".format(
                config=row["config_id"],
                status=cache["status"],
                read=cache["cache_read_tokens"],
                write=cache["cache_write_tokens"],
                detail=cache["detail"],
            )
        )
    balance_rows = _balance_summary(data)
    lines.extend(["", "## Balance Curves", ""])
    if balance_rows:
        lines.append("| config | completed traces | final balance mean | final balance min | minimum balance |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in balance_rows:
            lines.append(
                "| {config} | {count} | {final_mean} | {final_min} | {min_balance} |".format(
                    config=row["config_id"],
                    count=row["completed_traces"],
                    final_mean=_fmt(row["final_balance_mean"]),
                    final_min=_fmt(row["final_balance_min"]),
                    min_balance=_fmt(row["minimum_balance"]),
                )
            )
    else:
        lines.append("No completed traces were available for balance-curve summaries.")
    lines.extend(
        [
            "",
            "## Reliability",
            "",
            f"Completed cells: {data.summary['status_counts'].get('completed', 0)}",
            f"Failed cells: {data.summary['status_counts'].get('failed', 0)}",
            f"Budget-censored cells: {data.summary['status_counts'].get('failed_budget', 0)}",
            f"Skipped-budget cells: {data.summary['status_counts'].get('skipped_budget', 0)}",
            "",
        ]
    )
    if data.summary["failed_cells"]:
        lines.extend(["### Failed Cells", ""])
        lines.append("| model | status | error |")
        lines.append("|---|---|---|")
        for failure in data.summary["failed_cells"]:
            lines.append(
                "| {model} | {status} | {error} |".format(
                    model=failure["model"],
                    status=failure["status"],
                    error=_escape_table(failure["error"]),
                )
            )
        lines.append("")
    lines.extend(["## Money-Shot Traces", ""])
    for label, cell_id in data.summary["money_shots"].items():
        lines.append(f"- {label}: {cell_id or 'n/a'}")
    lines.append("")
    return "\n".join(lines)


def _model_note(row: dict[str, Any]) -> str:
    driver_label, driver_value = _dominant_loss(row)
    net = _fmt(row["net_revenue"]["mean"])
    net_ci = _fmt_ci(row["net_revenue"])
    fraction = _fmt(row["fraction_of_omniscient_optimal"]["mean"])
    manipulation = _fmt(row["manipulation_resistance_loss"]["mean"])
    jobs = _fmt(row["jobs_delivered"]["mean"])
    efficiency = _fmt(row["efficiency"]["mean"])
    reference_label = "upper-bound reference" if row["omniscient_reference_relaxation"] else "reactive optimum"
    return (
        f"- **{row['config_id']}** averages {net} net revenue (95% CI {net_ci}) and {fraction} of the {reference_label} "
        f"across {row['completed_cells']} completed cell(s). Its largest measured loss is {driver_label} "
        f"({_fmt(driver_value)}), with {manipulation} paired manipulation-resistance loss, {jobs} delivered jobs on average, "
        f"and {efficiency} fraction-of-optimal per compute dollar."
    )


def _dominant_loss(row: dict[str, Any]) -> tuple[str, Any]:
    candidates = [
        ("selection regret", row["selection_regret"]["mean"]),
        ("pricing regret", row["pricing_regret"]["mean"]),
        ("tool-selection regret", row["tool_selection_regret"]["mean"]),
        ("support concession", row["support_conceded_value"]["mean"]),
        ("manipulation-resistance loss", row["manipulation_resistance_loss"]["mean"]),
        ("coherence penalty", row["coherence_penalty"]["mean"]),
    ]
    delivery_pass = row["delivery_pass_rate"]["mean"]
    if delivery_pass is not None:
        candidates.append(("delivery failure rate", 1.0 - float(delivery_pass)))
    available = [(label, value) for label, value in candidates if value is not None]
    if not available:
        return ("unmeasured capability loss", None)
    return max(available, key=lambda item: abs(float(item[1])))


def _balance_summary(data: FindingsData) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Decimal]]] = {}
    for run in data.runs:
        balances = _trace_balances(run.trace_path)
        if not balances:
            continue
        grouped.setdefault(run.config_id, []).append(
            {
                "final": balances[-1],
                "minimum": min(balances),
            }
        )
    rows = []
    for config_id in sorted(grouped):
        points = grouped[config_id]
        finals = [point["final"] for point in points]
        minimums = [point["minimum"] for point in points]
        rows.append(
            {
                "config_id": config_id,
                "completed_traces": len(points),
                "final_balance_mean": float(sum(finals, Decimal("0")) / Decimal(len(finals))),
                "final_balance_min": float(min(finals)),
                "minimum_balance": float(min(minimums)),
            }
        )
    return rows


def _trace_balances(trace_path: Path) -> list[Decimal]:
    balances = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        balances.append(Decimal(str(event.get("balance_after", "0"))))
    return balances


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_ci(metric: dict[str, Any]) -> str:
    low = metric.get("ci95_low")
    high = metric.get("ci95_high")
    if low is None or high is None:
        return "n/a"
    return f"{_fmt(low)} to {_fmt(high)}"


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
