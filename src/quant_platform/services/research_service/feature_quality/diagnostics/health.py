"""Diagnostic health summaries for governed feature candidates."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from quant_platform.services.research_service.reports.statistics import mean, std_sample

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def build_feature_diagnostic_health(
    *,
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    direction_row: Mapping[str, object],
    official: Mapping[str, object],
    null: Mapping[str, object],
) -> dict[str, object]:
    """Summarise value health and gate metrics for one attribution feature."""
    grouped: dict[datetime, list[float]] = defaultdict(list)
    values: list[float] = []
    for sample in samples:
        raw = sample.features.get(feature_name)
        if raw is None:
            continue
        value = _float_value(raw, default=float("nan"))
        if not math.isfinite(value):
            continue
        values.append(value)
        grouped[sample.as_of].append(value)

    finite_count = len(values)
    nonzero_count = sum(1 for value in values if abs(value) >= 1e-12)
    zero_count = finite_count - nonzero_count
    daily_groups = len(grouped)
    active_date_count = sum(1 for day_values in grouped.values() if _has_nonzero(day_values))
    daily_unique = [_unique_ratio(day_values) for day_values in grouped.values()]
    metrics = _recommended_metrics(direction_row)
    ic_mean = _float_value(official.get("ic_mean"))
    null_p95 = _float_value(null.get("null_p95"))
    failed_gates = _recommended_list(direction_row, "failed_gates")
    health: dict[str, object] = {
        "sample_count": len(samples),
        "finite_count": finite_count,
        "nonzero_count": nonzero_count,
        "zero_count": zero_count,
        "nonzero_fraction": nonzero_count / max(1, finite_count),
        "zero_fraction": zero_count / max(1, finite_count),
        "active_date_count": active_date_count,
        "inactive_date_count": max(0, daily_groups - active_date_count),
        "daily_group_count": daily_groups,
        "dispersion": mean(
            [std_sample(day_values) for day_values in grouped.values() if len(day_values) >= 2]
        ),
        "unique_ratio": _unique_ratio(values),
        "mean_daily_unique_ratio": mean(daily_unique),
        "ic_mean": ic_mean,
        "null_p95": null_p95,
        "null_margin": ic_mean - null_p95,
        "icir": _float_value(metrics.get("icir")),
        "negative_ic_streak": _float_value(metrics.get("ic_negative_streak")),
        "cost_net_mean_return": _float_value(metrics.get("cost_net_mean_return")),
        "incremental_delta_ic": _float_value(metrics.get("incremental_delta_ic")),
        "failed_gates": failed_gates,
        "noise_failed": "noise" in failed_gates,
    }
    return health


def _recommended_metrics(row: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping_value(_recommended_orientation_payload(row).get("metrics"))


def _recommended_list(row: Mapping[str, object], key: str) -> list[str]:
    raw = _recommended_orientation_payload(row).get(key, ())
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(value) for value in raw]
    return []


def _recommended_orientation_payload(row: Mapping[str, object]) -> Mapping[str, object]:
    orientations = _mapping_value(row.get("orientations"))
    selected = orientations.get(str(row.get("recommended_orientation", "positive")), {})
    return _mapping_value(selected)


def _mapping_value(raw: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", raw) if isinstance(raw, Mapping) else {}


def _unique_ratio(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return len({round(value, 12) for value in finite}) / max(1, len(finite))


def _has_nonzero(values: Sequence[float]) -> bool:
    return any(abs(float(value)) >= 1e-12 for value in values if math.isfinite(float(value)))


def _float_value(raw: object, default: float = 0.0) -> float:
    if not isinstance(raw, int | float | str):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


__all__ = [
    "build_feature_diagnostic_health",
]
