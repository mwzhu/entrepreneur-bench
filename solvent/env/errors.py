class SolventError(Exception):
    """Base exception for Solvent environment errors."""


class UnknownJobError(SolventError):
    """Raised when a job id is not present in the market."""


class InvalidActionError(SolventError):
    """Raised when a tool call violates environment invariants."""


class AlreadyTerminatedError(SolventError):
    """Raised when a tool call is made after episode termination."""
