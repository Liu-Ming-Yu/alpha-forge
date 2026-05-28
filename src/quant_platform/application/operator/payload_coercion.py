"""Small JSON payload coercion helpers for application use cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def optional_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise TypeError(f"{name} must be a JSON object")


def optional_sequence(value: object, *, name: str) -> Sequence[object]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def require_float(value: object, *, name: str) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"{name} must be numeric")
