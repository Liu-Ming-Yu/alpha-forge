"""Typed coercion helpers for database rows and decoded JSON payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.engine import RowMapping


def require_datetime(row: Mapping[str, Any] | RowMapping, key: str) -> datetime:
    value = row[key]
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"{key} must be datetime-compatible, got {type(value).__name__}")


def optional_datetime(row: Mapping[str, Any] | RowMapping, key: str) -> datetime | None:
    value = row[key]
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"{key} must be datetime-compatible or None, got {type(value).__name__}")


def require_date(row: Mapping[str, Any] | RowMapping, key: str) -> date:
    value = row[key]
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"{key} must be date-compatible, got {type(value).__name__}")


def require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")


def optional_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if value is None:
        return {}
    return require_mapping(value, name=name)


def optional_string_mapping(value: object, *, name: str) -> Mapping[str, str]:
    return {key: str(item) for key, item in optional_mapping(value, name=name).items()}


def require_sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a non-string sequence, got {type(value).__name__}")
    return value


def optional_sequence(value: object, *, name: str) -> Sequence[object]:
    if value is None:
        return ()
    return require_sequence(value, name=name)


def require_float(value: object, *, name: str) -> float:
    if isinstance(value, int | float | str | Decimal):
        return float(value)
    raise TypeError(f"{name} must be float-compatible, got {type(value).__name__}")
