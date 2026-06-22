import json
from decimal import Decimal
from pathlib import Path

from solvent.cli.main import run_episode
from solvent.env.env import Environment
from solvent.env.models import EnvConfig
from solvent.env.pricing import TokenUsage
from solvent.harness.llm import LLMHarness
from solvent.harness.model_client import FakeClient, ModelResponse
from solvent.harness.stub import StubHarness
from solvent.scoring.scorecard import score_trace, scorecard_to_json
from solvent.viewer.trace_view import build_trace_view


def test_build_trace_view_extracts_public_timeline_artifact_and_scorecard(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    summary = run_episode(_config(trace_path, redteam_enabled=True), StubHarness("naive"))
    scorecard = score_trace(summary.trace_path)
    scorecard_path = tmp_path / "trace.scorecard.json"
    scorecard_path.write_text(scorecard_to_json(scorecard) + "\n", encoding="utf-8")

    view = build_trace_view(summary.trace_path, scorecard_path, root_dir=tmp_path)

    assert view["schema_version"] == "solvent_trace_view_v0_5"
    assert view["seed"] == 42
    assert view["config_id"] == "stub:test"
    assert view["redteam_enabled"] is True
    assert view["trace_path"] == "trace.jsonl"
    assert view["scorecard"]["trace_path"] == "trace.jsonl"
    assert len(view["balance_curve"]) == len(view["events"])
    assert view["jobs"]

    submitted = next(event for event in view["events"] if event["kind"] == "submitted")
    assert submitted["artifact_preview"].startswith("name,email,signup_date,plan")
    assert submitted["artifact_size"] >= len(submitted["artifact_preview"])
    assert len(submitted["artifact_sha256"]) == 64
    assert submitted["verify"]["passed"] is True
    assert submitted["verify"]["checks"]

    raw = json.dumps(view)
    assert "reservation_price" not in raw
    assert "est_cost" not in raw
    assert "rubric" not in raw
    assert "true_value" not in raw
    assert "is_decoy" not in raw
    assert "internal_difficulty" not in raw
    assert "pass_prob" not in raw
    assert str(tmp_path) not in raw


def test_trace_view_labels_v0_4_delivery_events(tmp_path: Path) -> None:
    env = Environment(
        EnvConfig(
            seed=42,
            config_id="tool:test",
            start_balance=Decimal("20.00"),
            horizon_ticks=1,
            overhead_per_tick=Decimal("0.05"),
            tool_call_cost=Decimal("0"),
            trace_path=tmp_path / "tool.jsonl",
            delivery_mode="tool_mediated",
        )
    )
    try:
        job = env.list_jobs()[0]
        env.bid(job.id, Decimal("0.50"))
        env.list_models()
        env.deliver(job.id, "tool-pro")
    finally:
        summary = env.finalize()

    view = build_trace_view(summary.trace_path, root_dir=tmp_path)
    titles = {event["kind"]: event["title"] for event in view["events"]}
    assert titles["models_listed"] == "Delivery models listed"
    assert titles["delivered"] == "Delivery attempted"
    assert titles["tool_price_charged"] == "Delivery tool charged"
    assert titles["delivery_passed"] == "Delivery passed"


def test_trace_view_hides_brain_metered_links_turns_and_diagnoses_delivery(tmp_path: Path) -> None:
    client = FakeClient(
        [
            ModelResponse({"name": "inspect_job", "arguments": {"job_id": "dc-42-0"}}, TokenUsage(80, 8), "Look at the job.", "Let me think about which job to take."),
            ModelResponse({"name": "bid", "arguments": {"job_id": "dc-42-0", "price": "0.50"}}, TokenUsage(70, 7), "Bid low."),
            ModelResponse({"name": "deliver", "arguments": {"job_id": "dc-42-0", "model": "tool-mini"}}, TokenUsage(90, 9), "Cheapest tool."),
            ModelResponse({"name": "end_tick", "arguments": {}}, TokenUsage(40, 4), "Done."),
        ]
    )
    trace_path = tmp_path / "agent.jsonl"
    summary = run_episode(
        EnvConfig(
            seed=42,
            config_id="claude-opus-4-8:base",
            start_balance=Decimal("20.00"),
            horizon_ticks=1,
            overhead_per_tick=Decimal("0.05"),
            tool_call_cost=Decimal("0"),
            trace_path=trace_path,
            delivery_mode="tool_mediated",
        ),
        LLMHarness(model="claude-opus-4-8", client=client),
    )

    view = build_trace_view(summary.trace_path, root_dir=tmp_path)

    # #3 brain_metered is gone from the timeline and the balance curve stays aligned.
    assert all(event["kind"] != "brain_metered" for event in view["events"])
    assert len(view["balance_curve"]) == len(view["events"])

    # #2 every model turn is ingested with its reasoning trace + message, and
    # events link back to their turn.
    assert [turn["tool"] for turn in view["turns"]] == ["inspect_job", "bid", "deliver", "end_tick"]
    assert view["turns"][0]["reasoning"] == "Let me think about which job to take."
    assert view["turns"][0]["message"] == "Look at the job."
    inspected = next(event for event in view["events"] if event["kind"] == "inspected")
    assert inspected["turn"] == 0
    assert next(event for event in view["events"] if event["kind"] == "episode_started")["turn"] is None

    # #1 the delivery carries an RNG-vs-model-choice diagnosis, read from the
    # ground truth the environment baked into the trace.
    raw_trace = summary.trace_path.read_text(encoding="utf-8")
    delivered_row = next(
        json.loads(line)
        for line in raw_trace.splitlines()
        if line.strip() and json.loads(line)["kind"] == "delivered"
    )
    ground_truth = delivered_row["payload"]["ground_truth"]
    assert ground_truth["internal_difficulty"] in {"easy", "med", "hard"}
    assert set(ground_truth["model_pass_probs"]) == {"tool-mini", "tool-mid", "tool-pro"}

    delivered = next(event for event in view["events"] if event["kind"] == "delivered")
    diagnosis = delivered["diagnosis"]
    assert diagnosis["model"] == "tool-mini"
    assert diagnosis["pass_prob"] == ground_truth["pass_prob"]
    assert diagnosis["verdict"] in {"passed", "unlucky_rng", "suboptimal_model"}
    assert any(model["chosen"] for model in diagnosis["models"])
    assert any(model["oracle_best"] for model in diagnosis["models"])
    assert view["delivery_summary"][0]["job_id"] == "dc-42-0"

    # The most overfit-able ground truth still never reaches trace or view.
    assert "reservation_price" not in raw_trace
    assert "is_decoy" not in raw_trace
    raw_view = json.dumps(view)
    assert "reservation_price" not in raw_view
    assert "true_value" not in raw_view


def _config(trace_path: Path, redteam_enabled: bool = False) -> EnvConfig:
    return EnvConfig(
        seed=42,
        config_id="stub:test",
        start_balance=Decimal("20.00"),
        horizon_ticks=5,
        overhead_per_tick=Decimal("0.05"),
        tool_call_cost=Decimal("0.01"),
        trace_path=trace_path,
        redteam_enabled=redteam_enabled,
        delivery_mode="direct",
    )
