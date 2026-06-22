import re
from decimal import Decimal
from pathlib import Path

from solvent.env.env import Environment
from solvent.env.models import EnvConfig
from solvent.scoring.scorecard import score_trace
from solvent.tasks.extract import generate_extract_job, verify_extract


def test_extract_generator_hides_answer_and_verifier_matches_planted_value() -> None:
    job = generate_extract_job(42, 0, "hard")
    answer = _answer_from_public_document(job.inputs["document"], job.inputs["field"])

    assert job.type == "extract"
    assert "answer" not in job.inputs
    assert verify_extract(job, answer).passed
    assert not verify_extract(job, "not the answer").passed


def test_extract_market_jobs_score_from_reconstructed_provenance(tmp_path: Path) -> None:
    env = Environment(
        EnvConfig(
            seed=42,
            config_id="extract:test",
            start_balance=Decimal("1000.00"),
            horizon_ticks=1,
            overhead_per_tick=Decimal("0.05"),
            tool_call_cost=Decimal("0"),
            trace_path=tmp_path / "extract.jsonl",
            delivery_mode="tool_mediated",
            task_mix={"extract": 1.0},
        )
    )
    try:
        job = env.list_jobs()[0]
        assert job.type == "extract"
        assert env.bid(job.id, Decimal("0.50"))["accepted"]
        env.deliver(job.id, "tool-pro")
        env.end_tick()
    finally:
        summary = env.finalize()

    scorecard = score_trace(summary.trace_path)
    assert scorecard.delivery.submitted_jobs == 1
    assert scorecard.coherence.dropped_jobs == 0


def _answer_from_public_document(document: str, field: str) -> str:
    if field == "invoice_id":
        return re.search(r"INV-\d{3}-\d{2}", document).group(0)
    if field == "total":
        match = re.search(r"(?:Final amount due: USD|Amount due: \$|Total: \$)(\d+\.\d{2})", document)
        return match.group(1)
    match = re.search(r"(?:Due date:|Payment due by|Invoice due date:)\s*(\d{4}-\d{2}-\d{2})", document)
    return match.group(1)
