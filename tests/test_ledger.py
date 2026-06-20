from decimal import Decimal

from solvent.env.ledger import Ledger


def test_ledger_debits_and_credits_exactly() -> None:
    ledger = Ledger(Decimal("20.00"))
    ledger.debit_burn(Decimal("0.01"))
    ledger.debit_overhead(Decimal("0.05"))
    ledger.credit_revenue(Decimal("0.50"))
    assert ledger.balance == Decimal("20.44")


def test_ledger_detects_insolvency() -> None:
    ledger = Ledger(Decimal("0.01"))
    assert not ledger.insolvent()
    ledger.debit_burn(Decimal("0.01"))
    assert ledger.insolvent()


def test_ledger_preserves_decimal_precision() -> None:
    ledger = Ledger(Decimal("0.30"))
    ledger.debit_burn(Decimal("0.10"))
    ledger.debit_burn(Decimal("0.10"))
    assert ledger.balance == Decimal("0.10")
