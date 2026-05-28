"""Shared bootstrap serialization helpers for operator-facing payloads."""

from __future__ import annotations

import enum
import uuid
from decimal import Decimal


def _json_default(obj: object) -> object:
    """Return a JSON-safe representation for common bootstrap payload values."""
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


__all__ = ["_json_default"]
