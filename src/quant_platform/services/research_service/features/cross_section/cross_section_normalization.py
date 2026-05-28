"""Pure cross-sectional normalization utilities."""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import TypeVar

import structlog

K = TypeVar("K", bound=Hashable)

log = structlog.get_logger(__name__)


def rank_normalize(values: dict[K, float]) -> dict[K, float]:
    """Cross-sectional percentile rank mapped linearly to [-1, 1]."""
    if not values:
        return {}
    n = len(values)
    if n == 1:
        return {k: 0.0 for k in values}

    sorted_items = sorted(values.items(), key=lambda kv: kv[1])
    result: dict[K, float] = {}
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_items[j][1] == sorted_items[j + 1][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        normalized = 2.0 * avg_rank / (n - 1) - 1.0
        for k in range(i, j + 1):
            result[sorted_items[k][0]] = normalized
        i = j + 1
    return result


def winsorize(
    values: dict[K, float],
    lower_pct: float = 0.05,
    upper_pct: float = 0.05,
) -> dict[K, float]:
    """Clip outliers to the lower/upper percentile boundaries."""
    if not values or (lower_pct <= 0 and upper_pct <= 0):
        return dict(values)
    sorted_vals = sorted(values.values())
    n = len(sorted_vals)
    lo_idx = max(0, int(math.floor(lower_pct * n)))
    hi_idx = min(n - 1, int(math.ceil((1.0 - upper_pct) * n)) - 1)
    lo = sorted_vals[lo_idx]
    hi = sorted_vals[hi_idx]
    return {k: max(lo, min(hi, v)) for k, v in values.items()}


def z_score_normalize(
    values: dict[K, float],
    factor_name: str = "",
) -> dict[K, float]:
    """Z-score normalize across the cross-section."""
    if not values:
        return {}
    vals = list(values.values())
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    if variance <= 0:
        log.warning(
            "cross_section.flat_factor_detected",
            factor_name=factor_name or "<unknown>",
            n_instruments=len(vals),
        )
        return {k: 0.0 for k, _ in values.items()}
    std = math.sqrt(variance)
    return {k: (v - mean) / std for k, v in values.items()}


def blend_factors(
    factor_scores: list[dict[K, float]],
    weights: list[float] | None = None,
) -> dict[K, float]:
    """Weighted blend of multiple normalized factor dicts."""
    if not factor_scores:
        return {}

    all_keys: set[K] = set()
    for factor_score in factor_scores:
        all_keys.update(factor_score.keys())

    n = len(factor_scores)
    w = weights if weights is not None else [1.0 / n] * n
    total_w = sum(w)
    if total_w == 0:
        return {k: 0.0 for k in all_keys}

    result: dict[K, float] = {}
    for k in all_keys:
        composite = sum(wi * fs.get(k, 0.0) for wi, fs in zip(w, factor_scores, strict=False))
        result[k] = composite / total_w
    return result
