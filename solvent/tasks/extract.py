from __future__ import annotations

import random
import re
from decimal import Decimal

from solvent.env.models import Check, CheckResult, Job, Rubric, VerifyResult


def generate_extract_job(seed: int, job_index: int, difficulty: str = "easy") -> Job:
    rng = random.Random(f"{seed}:{job_index}:extract:{difficulty}")
    customer = rng.choice(["Northstar Labs", "Blue Finch Supply", "Arbor Analytics"])
    invoice_id = f"INV-{seed % 1000:03d}-{job_index:02d}"
    total = (Decimal(rng.randrange(12000, 78000)) / Decimal("100")).quantize(Decimal("0.01"))
    due_date = f"2026-0{rng.randrange(7, 10)}-{rng.randrange(10, 28)}"
    document = _document(customer, invoice_id, total, due_date, difficulty)
    field = rng.choice(["invoice_id", "total", "due_date"])
    answer = {"invoice_id": invoice_id, "total": str(total), "due_date": due_date}[field]
    reservation_price = (Decimal("0.85") + Decimal(rng.randrange(10, 76)) / Decimal("100")).quantize(Decimal("0.01"))

    return Job(
        id=f"ex-{seed}-{job_index}",
        type="extract",
        brief=f"Extract the {field} from the document. Return only the value, with no explanation.",
        inputs={"document": document, "field": field},
        arrival_tick=job_index,
        reservation_price=reservation_price,
        est_cost=Decimal("0.20"),
        rubric=_rubric(answer),
        internal_difficulty=difficulty,
    )


def verify_extract(job: Job, artifact: str) -> VerifyResult:
    return VerifyResult([check.predicate(artifact, job.inputs) for check in job.rubric.checks])


def _rubric(answer: str) -> Rubric:
    def check_exact_or_fuzzy(artifact: str, inputs: dict) -> CheckResult:
        normalized_artifact = _normalize(artifact)
        normalized_answer = _normalize(answer)
        passed = normalized_artifact == normalized_answer
        return CheckResult(
            "planted_answer",
            passed,
            "answer matched" if passed else f"expected {answer}",
        )

    return Rubric(checks=(Check("planted_answer", check_exact_or_fuzzy),))


def _document(customer: str, invoice_id: str, total: Decimal, due_date: str, difficulty: str) -> str:
    if difficulty == "easy":
        return f"Invoice {invoice_id}\nCustomer: {customer}\nTotal: ${total}\nDue date: {due_date}\n"
    if difficulty == "med":
        return (
            f"Customer memo for {customer}\n"
            f"Reference number: PO-{invoice_id[-2:]}\n"
            f"Invoice ID: {invoice_id}\n"
            f"Subtotal: ${total - Decimal('12.00')}\n"
            f"Tax and fees: $12.00\n"
            f"Amount due: ${total}\n"
            f"Payment due by {due_date}.\n"
        )
    return (
        f"{customer} account packet\n"
        f"Previous invoice INV-000-99 was voided.\n"
        f"Current invoice identifier: {invoice_id}.\n"
        f"Budget note: do not confuse approved spend ${total + Decimal('40.00')} with the invoice total.\n"
        f"Final amount due: USD {total}.\n"
        f"Renewal reminder date: 2026-12-31.\n"
        f"Invoice due date: {due_date}.\n"
    )


def _normalize(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"^\$|^usd\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
