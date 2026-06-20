from decimal import Decimal


class Ledger:
    """Exact money ledger for burn, overhead, and revenue."""

    def __init__(self, start_balance: Decimal):
        self._start_balance = start_balance
        self._balance = start_balance

    @property
    def start_balance(self) -> Decimal:
        return self._start_balance

    @property
    def balance(self) -> Decimal:
        return self._balance

    def debit_burn(self, amount: Decimal) -> Decimal:
        self._balance -= amount
        return self._balance

    def debit_overhead(self, amount: Decimal) -> Decimal:
        self._balance -= amount
        return self._balance

    def credit_revenue(self, amount: Decimal) -> Decimal:
        self._balance += amount
        return self._balance

    def insolvent(self) -> bool:
        return self._balance <= Decimal("0")
