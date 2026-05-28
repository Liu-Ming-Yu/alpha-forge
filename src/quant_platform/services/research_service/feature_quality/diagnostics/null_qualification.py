"""Null-baseline qualification helpers for diagnostic feature attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast


def null_qualified_features(
    attribution: Mapping[str, object],
    *,
    official_horizon_days: int,
) -> dict[str, tuple[str, ...]]:
    """Classify features by whether their official IC beats deterministic null p95."""
    horizon_key = str(official_horizon_days)
    audited: list[str] = []
    qualified: list[str] = []
    quarantined: list[str] = []
    for row in _feature_rows(attribution):
        name = str(row.get("feature_name", ""))
        if not name:
            continue
        audited.append(name)
        official = _mapping_value(row.get("horizon_comparison", {})).get(horizon_key, {})
        official_row = _mapping_value(official)
        null = _mapping_value(row.get("null_baseline", {}))
        observations = _float_value(official_row.get("ic_observations", 0.0))
        ic_mean = _float_value(official_row.get("ic_mean", 0.0))
        null_p95 = _float_value(null.get("null_p95", 0.0))
        if observations >= 252.0 and ic_mean > null_p95:
            qualified.append(name)
        else:
            quarantined.append(name)
    return {
        "audited": tuple(audited),
        "qualified": tuple(qualified),
        "quarantined": tuple(quarantined),
    }


def _feature_rows(payload: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = payload.get("features", ())
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return ()
    return tuple(cast("Mapping[str, object]", row) for row in raw if isinstance(row, Mapping))


def _mapping_value(raw: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", raw) if isinstance(raw, Mapping) else {}


def _float_value(raw: object, default: float = 0.0) -> float:
    if not isinstance(raw, int | float | str):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError, OverflowError):
        return default


__all__ = ["null_qualified_features"]
