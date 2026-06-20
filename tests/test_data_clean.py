from solvent.tasks.data_clean import build_clean_csv, generate_data_clean_job, verify_data_clean


def test_generated_job_is_stable_for_same_seed_and_index() -> None:
    first = generate_data_clean_job(42, 0)
    second = generate_data_clean_job(42, 0)
    assert first.id == second.id
    assert first.inputs == second.inputs
    assert first.reservation_price == second.reservation_price


def test_valid_artifact_passes_all_checks() -> None:
    job = generate_data_clean_job(42, 0)
    result = verify_data_clean(job, build_clean_csv(job.inputs["csv"]))
    assert result.passed
    assert result.score == 1.0


def test_malformed_csv_fails_parse_check() -> None:
    job = generate_data_clean_job(42, 0)
    result = verify_data_clean(job, '"unterminated\n')
    assert not result.passed
    assert _check(result, "csv_parses").passed is False


def test_wrong_columns_fail_schema_check() -> None:
    job = generate_data_clean_job(42, 0)
    result = verify_data_clean(job, "email,name,signup_date,plan\nada@example.com,Ada,2026-01-04,pro\n")
    assert not result.passed
    assert _check(result, "schema").passed is False


def test_blank_required_value_fails_completeness_check() -> None:
    job = generate_data_clean_job(42, 0)
    result = verify_data_clean(job, "name,email,signup_date,plan\nAda,,2026-01-04,pro\n")
    assert not result.passed
    assert _check(result, "no_blank_required_cells").passed is False


def test_unnormalized_email_fails_email_check() -> None:
    job = generate_data_clean_job(42, 0)
    result = verify_data_clean(job, "name,email,signup_date,plan\nAda,ADA@EXAMPLE.COM,2026-01-04,pro\n")
    assert not result.passed
    assert _check(result, "emails_lowercase").passed is False


def _check(result, name: str):
    return next(check for check in result.checks if check.name == name)
