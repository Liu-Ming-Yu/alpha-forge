"""Small presentation helpers for CLI command handlers."""

from __future__ import annotations

import enum
import json
import uuid
from decimal import Decimal
from typing import Any

from quant_platform.application.results import ResultPresentation, UseCaseResult


def json_default(obj: object) -> object:
    """Return a JSON-safe representation for common domain values."""
    if isinstance(obj, (uuid.UUID, Decimal)):
        return str(obj)
    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        value = isoformat()
        if isinstance(value, str):
            return value
    if isinstance(obj, enum.Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def print_json(payload: object) -> None:
    """Print an operator-facing JSON payload with stable formatting."""
    print(json.dumps(payload, default=json_default, indent=2, sort_keys=True))


def print_json_or_exit(
    payload: dict[str, Any],
    *,
    fail_on_blocked: bool,
    exit_code: int = 2,
) -> None:
    """Print a JSON payload and exit when a command explicitly asserts success."""
    print_json(payload)
    if fail_on_blocked and not bool(payload.get("passed", False)):
        raise SystemExit(exit_code)


def render_result(result: UseCaseResult[Any]) -> int:
    """Render an application result and return its process exit code."""
    if result.presentation == ResultPresentation.JSON:
        print_json(result.payload)
    elif result.presentation == ResultPresentation.KEY_VALUE:
        payload = result.payload or {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                print(f"  {key}: {value}")
        else:
            print(str(payload))
    elif result.presentation == ResultPresentation.TEXT:
        if result.message:
            print(result.message)
        elif result.payload is not None:
            print(str(result.payload))
    return int(result.exit_code)


__all__ = [
    "json_default",
    "print_json",
    "print_json_or_exit",
    "render_result",
]
