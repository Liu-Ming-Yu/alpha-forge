"""Small numeric helpers for text-candidate formulas."""

from __future__ import annotations

from typing import Any, cast


def _is_finite_number(raw: object) -> bool:
    try:
        value = float(cast("Any", raw))
    except (TypeError, ValueError, OverflowError):
        return False
    return value == value and value not in {float("inf"), float("-inf")}


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0
