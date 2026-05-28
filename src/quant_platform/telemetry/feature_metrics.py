"""Feature-distribution telemetry helpers.

Pure telemetry: computes per-feature mean/std/NaN gauges and EMA-based mean-shift
detection over feature rows. Lives in the telemetry layer so both the
research-service compute path and the data-service maintenance scheduler can
emit feature-distribution metrics without a cross-service import.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import structlog

from quant_platform.telemetry.metrics import (
    record_feature_nan,
    set_feature_mean,
    set_feature_std,
)

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

_ema_state: dict[str, tuple[float, float]] = {}
_EMA_ALPHA = 0.1


def emit_feature_distribution_metrics(
    merged: Mapping[Any, Mapping[str, float]],
    feature_set_version: str,
    mean_shift_threshold: float = 2.0,
) -> None:
    """Compute per-feature statistics, emit gauges, and detect mean shifts."""
    by_feature: dict[str, list[float]] = {}
    for features in merged.values():
        for name, value in features.items():
            if name.startswith("_"):
                continue
            if not math.isfinite(value):
                record_feature_nan(name, feature_set_version)
                continue
            by_feature.setdefault(name, []).append(value)

    for name, values in by_feature.items():
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
        std = math.sqrt(variance)

        set_feature_mean(name, feature_set_version, mean)
        set_feature_std(name, feature_set_version, std)

        prev_ema_mean, prev_ema_var = _ema_state.get(name, (mean, variance or 1.0))
        ema_mean = (1 - _EMA_ALPHA) * prev_ema_mean + _EMA_ALPHA * mean
        ema_var = (1 - _EMA_ALPHA) * prev_ema_var + _EMA_ALPHA * (mean - prev_ema_mean) ** 2
        _ema_state[name] = (ema_mean, ema_var)

        ema_std = math.sqrt(max(ema_var, 1e-12))
        z = abs(mean - prev_ema_mean) / ema_std
        if z > mean_shift_threshold:
            log.warning(
                "feature_pipeline.mean_shift_detected",
                feature_name=name,
                feature_set_version=feature_set_version,
                current_mean=mean,
                ema_mean=prev_ema_mean,
                z_score=z,
                threshold=mean_shift_threshold,
            )


__all__ = ["emit_feature_distribution_metrics"]
