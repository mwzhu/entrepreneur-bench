from decimal import Decimal
from pathlib import Path

from solvent.cli.main import run_episode
from solvent.delivery.menu import DeliveryMenu
from solvent.env.models import EnvConfig, Job, Rubric
from solvent.harness.stub import StubHarness
from solvent.scoring.events import AcceptedJobFact, TraceFacts
from solvent.scoring.scorecard import ScorecardBuilder
from solvent.scoring.scorecard import score_trace


def test_bad_delivery_scores_delivery_failure_without_type_error(tmp_path: Path) -> None:
    summary = run_episode(_config(tmp_path / "bad.jsonl"), StubHarness("bad_delivery"))
    scorecard = score_trace(summary.trace_path)
    assert scorecard.delivery.pass_rate == 0.0
    assert scorecard.gross_score == Decimal("0.00")
    assert scorecard.fraction_of_realizable is None


def test_underprice_creates_pricing_regret_overprice_creates_selection_regret(tmp_path: Path) -> None:
    under = run_episode(_config(tmp_path / "under.jsonl"), StubHarness("underprice"))
    over = run_episode(_config(tmp_path / "over.jsonl"), StubHarness("overprice"))
    under_card = score_trace(under.trace_path)
    over_card = score_trace(over.trace_path)
    # Underpricing accepts good jobs far below the hidden ceiling: surplus left on the table.
    assert under_card.pricing.counter_accepts > 0
    assert under_card.pricing.surplus_left > 0
    assert under_card.pricing.pricing_regret > 0
    # Overpricing counters above every ceiling: counters are rejected, nothing is
    # accepted, so the loss is selection (good jobs walked away from), not pricing.
    assert over_card.pricing.rejected_counters > 0
    assert over_card.pricing.accepted_jobs == 0
    assert over_card.pricing.pricing_regret == Decimal("0.00")
    assert over_card.selection.selection_regret > 0


def test_decoy_chaser_and_invalid_loop_surface_stage_signals(tmp_path: Path) -> None:
    decoy = run_episode(_config(tmp_path / "decoy.jsonl"), StubHarness("decoy_chaser"))
    invalid = run_episode(_config(tmp_path / "invalid.jsonl"), StubHarness("invalid_loop"))
    decoy_card = score_trace(decoy.trace_path)
    invalid_card = score_trace(invalid.trace_path)
    assert decoy_card.selection.decoys_chosen == 2
    assert decoy_card.selection.precision is not None
    assert invalid_card.coherence.duplicate_bid_attempts >= 3
    assert invalid_card.coherence.coherence_penalty > 0


def test_selection_regret_does_not_penalize_alternative_overlapping_optimum(tmp_path: Path) -> None:
    jobs = [
        _overlapping_job("job-a"),
        _overlapping_job("job-b"),
    ]
    facts = _trace_facts_for_selection(
        accepted_jobs={
            "job-a": AcceptedJobFact(
                job_id="job-a",
                contract_price=Decimal("80.00"),
                starting_price=Decimal("80.00"),
                via_counter=False,
            )
        }
    )

    selection = ScorecardBuilder(tmp_path / "overlap.jsonl", jobs, facts)._selection()

    assert selection.good_available == 2
    assert selection.good_chosen == 1
    assert selection.selection_regret == Decimal("0.00")


def test_redteam_naive_concedes_and_procedure_resists(tmp_path: Path) -> None:
    naive = run_episode(_config(tmp_path / "naive.jsonl", redteam_enabled=True), StubHarness("naive"))
    procedure = run_episode(_config(tmp_path / "procedure.jsonl", redteam_enabled=True), StubHarness("procedure"))
    naive_card = score_trace(naive.trace_path)
    procedure_card = score_trace(procedure.trace_path)
    assert naive_card.support.manipulation_conceded == 1
    assert naive_card.support.conceded_value > 0
    assert procedure_card.support.manipulation_resisted == 1
    assert procedure_card.support.manipulation_conceded == 0


def test_unresolved_manipulation_does_not_count_as_resistance_failure(tmp_path: Path) -> None:
    from solvent.env.env import Environment

    env = Environment(_config(tmp_path / "unresolved.jsonl", redteam_enabled=True))
    try:
        job = env.list_jobs()[0]
        assert env.bid(job.id, Decimal("0.50"))["manipulation"] is not None
    finally:
        summary = env.finalize()
    scorecard = score_trace(summary.trace_path)
    assert scorecard.support.manipulation_attempts == 1
    assert scorecard.support.resistance_rate is None
    assert scorecard.coherence.dropped_jobs == 1


def test_v0_1_trace_without_market_metadata_can_be_scored(tmp_path: Path) -> None:
    summary = run_episode(
        _config(tmp_path / "v1.jsonl", market_version="data_clean_static_v0_1", market_size=3, decoy_rate=Decimal("0")),
        StubHarness("happy_path"),
    )
    text = summary.trace_path.read_text(encoding="utf-8")
    text = text.replace(',"market_version":"data_clean_static_v0_1"', "")
    text = text.replace(',"market_size":3', "")
    text = text.replace(',"decoy_rate":"0"', "")
    text = text.replace(',"redteam_enabled":false', "")
    text = text.replace(',"horizon_ticks":5', "")
    text = text.replace(',"overhead_per_tick":"0.05"', "")
    text = text.replace(',"tool_call_cost":"0.01"', "")
    legacy_path = tmp_path / "legacy.jsonl"
    legacy_path.write_text(text, encoding="utf-8")
    scorecard = score_trace(legacy_path)
    assert scorecard.seed == 42
    assert scorecard.selection.good_available == 3
    assert scorecard.compatibility_estimated_horizon is True


def _overlapping_job(job_id: str) -> Job:
    return Job(
        id=job_id,
        type="data_clean",
        brief="Overlapping job",
        inputs={"csv": "name,email,signup_date,plan\n"},
        arrival_tick=0,
        arrival_minute=0,
        expiry_minute=1,
        reservation_price=Decimal("100.00"),
        starting_price=Decimal("80.00"),
        est_cost=Decimal("10.00"),
        rubric=Rubric(checks=()),
        internal_difficulty="easy",
    )


def _trace_facts_for_selection(*, accepted_jobs: dict[str, AcceptedJobFact]) -> TraceFacts:
    return TraceFacts(
        seed=1,
        config_id="test:selection",
        start_balance=Decimal("1000.00"),
        horizon_ticks=1,
        horizon_minutes=1,
        overhead_per_tick=Decimal("0"),
        overhead_per_minute=Decimal("0"),
        tool_call_cost=Decimal("0"),
        market_version="business_stream_v0_5",
        market_size=2,
        arrival_rate_per_day=Decimal("2"),
        job_ttl_minutes=1,
        decoy_rate=Decimal("0"),
        redteam_enabled=False,
        end_balance=Decimal("1000.00"),
        net_revenue=Decimal("0"),
        tool_burn=Decimal("0"),
        overhead_burn=Decimal("0"),
        bids=[],
        submissions=[],
        accepted_jobs=accepted_jobs,
        declined_jobs=set(),
        counter_rejected_jobs=set(),
        paid_revenue_by_job={},
        manipulation_attempts=set(),
        manipulation_conceded={},
        manipulation_resisted=set(),
        invalid_actions=[],
        terminated_reason="horizon",
        compatibility_estimated_horizon=False,
        delivery_mode="tool_mediated",
        menu_checksum=DeliveryMenu.load_default().checksum,
        task_mix={"data_clean": 1.0},
        difficulty_distribution={"easy": 1.0},
        delivery_attempts=[],
    )


def _config(
    trace_path: Path,
    redteam_enabled: bool = False,
    market_version: str = "data_clean_static_v0_2",
    market_size: int = 5,
    decoy_rate: Decimal = Decimal("0.40"),
) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="stub:test",
        start_balance=Decimal("20.00"),
        horizon_ticks=5,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0.01"),
        trace_path=trace_path,
        market_version=market_version,
        market_size=market_size,
        decoy_rate=decoy_rate,
        redteam_enabled=redteam_enabled,
        delivery_mode="direct",
    )
