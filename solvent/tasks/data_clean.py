from __future__ import annotations

import csv
import io
import random
from decimal import Decimal
from typing import Any

from solvent.env.models import Check, CheckResult, Job, Rubric, VerifyResult

REQUIRED_COLUMNS = ["name", "email", "signup_date", "plan"]
SOURCE_COLUMNS = ["full_name", "email", "signup_date", "plan"]
VALID_PLANS = {"free", "pro", "enterprise"}


def generate_data_clean_job(seed: int, job_index: int) -> Job:
    rng = random.Random(f"{seed}:{job_index}:data_clean")
    people = [
        (" Ada Lovelace ", "ADA@EXAMPLE.COM", "2026-01-04", " Pro"),
        ("Grace Hopper", "grace@example.com", "2026-01-05", "enterprise"),
        (" Katherine Johnson ", "KATHERINE@EXAMPLE.COM", "2026-01-07", "FREE "),
        ("Barbara Liskov", "BARBARA@EXAMPLE.COM", "2026-01-08", "pro"),
        ("Missing Email", "", "2026-01-06", "free"),
        ("Invalid Plan", "invalid@example.com", "2026-01-09", "trial"),
    ]
    rng.shuffle(people)
    rows = people[:4]
    if not any(row[1] == "" for row in rows):
        rows[-1] = ("Missing Email", "", "2026-01-06", "free")
    if not any(row[3].strip().lower() == "trial" for row in rows):
        rows[-2] = ("Invalid Plan", "invalid@example.com", "2026-01-09", "trial")

    raw_csv = _write_csv(SOURCE_COLUMNS, rows)
    cents = Decimal(rng.randrange(25, 76)) / Decimal("100")
    reservation_price = Decimal("0.75") + cents
    est_cost = Decimal("0.20")

    brief = (
        "Clean the customer signup CSV. Return CSV text only with columns "
        "name,email,signup_date,plan in that exact order. Trim names, lowercase "
        "emails and plan values, keep plans only when they are free, pro, or "
        "enterprise, and drop rows with blank required cells."
    )
    return Job(
        id=f"dc-{seed}-{job_index}",
        type="data_clean",
        brief=brief,
        inputs={"csv": raw_csv, "required_columns": list(REQUIRED_COLUMNS)},
        arrival_tick=job_index,
        reservation_price=reservation_price.quantize(Decimal("0.01")),
        est_cost=est_cost,
        rubric=_rubric(),
    )


def verify_data_clean(job: Job, artifact: str) -> VerifyResult:
    return VerifyResult([check.predicate(artifact, job.inputs) for check in job.rubric.checks])


def build_clean_csv(raw_csv: str) -> str:
    parsed = _parse_csv(raw_csv)
    output_rows = []
    for row in parsed["rows"]:
        cleaned = {
            "name": row["full_name"].strip(),
            "email": row["email"].strip().lower(),
            "signup_date": row["signup_date"].strip(),
            "plan": row["plan"].strip().lower(),
        }
        if all(cleaned.values()) and cleaned["plan"] in VALID_PLANS:
            output_rows.append([cleaned[column] for column in REQUIRED_COLUMNS])
    return _write_csv(REQUIRED_COLUMNS, output_rows)


def _rubric() -> Rubric:
    return Rubric(
        checks=(
            Check("csv_parses", _check_csv_parses),
            Check("schema", _check_schema),
            Check("row_count", _check_row_count),
            Check("no_blank_required_cells", _check_no_blank_required_cells),
            Check("emails_lowercase", _check_emails_lowercase),
            Check("names_trimmed", _check_names_trimmed),
            Check("plan_values", _check_plan_values),
            Check("no_extra_columns", _check_no_extra_columns),
        )
    )


def _check_csv_parses(artifact: str, inputs: dict[str, Any]) -> CheckResult:
    try:
        _parse_csv(artifact)
        return CheckResult("csv_parses", True, "CSV parsed successfully")
    except csv.Error as exc:
        return CheckResult("csv_parses", False, f"CSV parse error: {exc}")


def _check_schema(artifact: str, inputs: dict[str, Any]) -> CheckResult:
    parsed = _parse_artifact_or_none(artifact)
    if parsed is None:
        return CheckResult("schema", False, "CSV did not parse")
    passed = parsed["columns"] == REQUIRED_COLUMNS
    return CheckResult("schema", passed, "required columns match" if passed else "columns must be name,email,signup_date,plan")


def _check_row_count(artifact: str, inputs: dict[str, Any]) -> CheckResult:
    parsed = _parse_artifact_or_none(artifact)
    if parsed is None:
        return CheckResult("row_count", False, "CSV did not parse")
    expected = len(_expected_rows(inputs["csv"]))
    actual = len(parsed["rows"])
    return CheckResult("row_count", actual == expected, f"expected {expected} valid rows, found {actual}")


def _check_no_blank_required_cells(artifact: str, inputs: dict[str, Any]) -> CheckResult:
    parsed = _parse_artifact_or_none(artifact)
    if parsed is None:
        return CheckResult("no_blank_required_cells", False, "CSV did not parse")
    blanks = [
        row_number
        for row_number, row in enumerate(parsed["rows"], start=2)
        if any(str(row.get(column, "")).strip() == "" for column in REQUIRED_COLUMNS)
    ]
    return CheckResult("no_blank_required_cells", not blanks, f"blank required cells on rows {blanks}" if blanks else "no blank required cells")


def _check_emails_lowercase(artifact: str, inputs: dict[str, Any]) -> CheckResult:
    parsed = _parse_artifact_or_none(artifact)
    if parsed is None:
        return CheckResult("emails_lowercase", False, "CSV did not parse")
    bad = [row["email"] for row in parsed["rows"] if row.get("email", "") != row.get("email", "").lower()]
    return CheckResult("emails_lowercase", not bad, f"emails not lowercased: {bad}" if bad else "emails are lowercased")


def _check_names_trimmed(artifact: str, inputs: dict[str, Any]) -> CheckResult:
    parsed = _parse_artifact_or_none(artifact)
    if parsed is None:
        return CheckResult("names_trimmed", False, "CSV did not parse")
    bad = [row["name"] for row in parsed["rows"] if row.get("name", "") != row.get("name", "").strip()]
    return CheckResult("names_trimmed", not bad, f"names not trimmed: {bad}" if bad else "names are trimmed")


def _check_plan_values(artifact: str, inputs: dict[str, Any]) -> CheckResult:
    parsed = _parse_artifact_or_none(artifact)
    if parsed is None:
        return CheckResult("plan_values", False, "CSV did not parse")
    bad = [row.get("plan", "") for row in parsed["rows"] if row.get("plan", "") not in VALID_PLANS]
    return CheckResult("plan_values", not bad, f"invalid plans: {bad}" if bad else "plans are valid")


def _check_no_extra_columns(artifact: str, inputs: dict[str, Any]) -> CheckResult:
    parsed = _parse_artifact_or_none(artifact)
    if parsed is None:
        return CheckResult("no_extra_columns", False, "CSV did not parse")
    passed = len(parsed["columns"]) == len(REQUIRED_COLUMNS)
    return CheckResult("no_extra_columns", passed, "no extra columns" if passed else "extra columns present")


def _expected_rows(raw_csv: str) -> list[dict[str, str]]:
    return _parse_csv(build_clean_csv(raw_csv))["rows"]


def _parse_artifact_or_none(artifact: str) -> dict[str, Any] | None:
    try:
        return _parse_csv(artifact)
    except csv.Error:
        return None


def _parse_csv(text: str) -> dict[str, Any]:
    stream = io.StringIO(text)
    reader = csv.reader(stream, strict=True)
    try:
        rows = list(reader)
    except csv.Error:
        raise
    if not rows:
        raise csv.Error("empty CSV")
    columns = rows[0]
    data_rows = []
    for row in rows[1:]:
        if len(row) != len(columns):
            raise csv.Error("row has different number of columns than header")
        data_rows.append(dict(zip(columns, row)))
    return {"columns": columns, "rows": data_rows}


def _write_csv(columns: list[str], rows: list[tuple[str, ...]] | list[list[str]]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue()
