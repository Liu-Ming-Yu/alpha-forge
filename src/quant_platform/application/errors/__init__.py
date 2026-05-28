"""Operator-facing application errors."""

from __future__ import annotations


class OperatorUsageError(ValueError):
    """Raised when an operator supplies invalid command input.

    Subclasses :class:`ValueError` so the failure still reads as bad input even
    if it escapes the CLI boundary. The CLI catches it and renders a clean
    message with a non-zero exit code instead of a traceback.
    """


__all__ = ["OperatorUsageError"]
