"""Pure research statistics helpers.

The research and governance paths intentionally share these small utilities so
IC, rank correlation, bootstrap intervals, and streak logic do not drift across
feature audits, campaigns, boosted-model validation, and shadow scoring.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean, or ``0.0`` for an empty iterable."""
    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


def std_sample(values: Sequence[float]) -> float:
    """Return sample standard deviation, or ``0.0`` for fewer than two values."""
    if len(values) < 2:
        return 0.0
    value_mean = mean(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / (len(values) - 1))


def pearson(
    x: Sequence[float],
    y: Sequence[float],
    *,
    invalid_value: float = 0.0,
    constant_value: float = 0.0,
) -> float:
    """Return Pearson correlation with explicit invalid/constant fallbacks."""
    if len(x) < 2 or len(x) != len(y):
        return invalid_value
    x_mean = mean(x)
    y_mean = mean(y)
    x_dev = [value - x_mean for value in x]
    y_dev = [value - y_mean for value in y]
    denom = math.sqrt(sum(value * value for value in x_dev) * sum(value * value for value in y_dev))
    if denom <= 0:
        return constant_value
    return sum(a * b for a, b in zip(x_dev, y_dev, strict=True)) / denom


def average_ranks(values: Sequence[float]) -> list[float]:
    """Return zero-based average ranks with classic tie handling."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = rank
        i = j + 1
    return ranks


def spearman_ic(
    values: Sequence[float],
    labels: Sequence[float],
    *,
    invalid_value: float = 0.0,
    constant_value: float = 0.0,
    drop_non_finite: bool = False,
) -> float:
    """Return cross-sectional Spearman rank IC with configurable fallbacks."""
    if len(values) != len(labels):
        return invalid_value
    paired = [(float(value), float(label)) for value, label in zip(values, labels, strict=True)]
    if drop_non_finite:
        paired = [
            (value, label)
            for value, label in paired
            if math.isfinite(value) and math.isfinite(label)
        ]
    if len(paired) < 2:
        return invalid_value
    ranked_values = average_ranks([value for value, _ in paired])
    ranked_labels = average_ranks([label for _, label in paired])
    return pearson(
        ranked_values,
        ranked_labels,
        invalid_value=invalid_value,
        constant_value=constant_value,
    )


def rolling_mean(values: Sequence[float], window: int) -> float:
    """Return the mean of the latest ``window`` observations."""
    if not values:
        return 0.0
    return mean(values[-min(window, len(values)) :])


def negative_streak(values: Sequence[float]) -> int:
    """Return the longest consecutive run of negative values."""
    current = 0
    worst = 0
    for value in values:
        if value < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 500,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    round_indices: bool = False,
) -> tuple[float, float]:
    """Return a bootstrap confidence interval for the sample mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(max(1, samples)):
        means.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    means.sort()

    def _index(q: float) -> int:
        raw = q * (len(means) - 1)
        idx = int(round(raw)) if round_indices else int(raw)
        return min(len(means) - 1, max(0, idx))

    return means[_index(lower_quantile)], means[_index(upper_quantile)]


def lag1_autocorr(values: Sequence[float]) -> float:
    """Return one-period autocorrelation, or ``0.0`` for short inputs."""
    if len(values) < 3:
        return 0.0
    return pearson(values[:-1], values[1:])


def winsor_impact(values: Sequence[float]) -> float:
    """Return mean absolute 5/95 winsorization impact scaled by mean abs value."""
    if len(values) < 20:
        return 0.0
    ordered = sorted(values)
    lo = ordered[int(0.05 * (len(ordered) - 1))]
    hi = ordered[int(0.95 * (len(ordered) - 1))]
    clipped = [max(lo, min(hi, value)) for value in values]
    denom = mean([abs(value) for value in values]) or 1.0
    return mean([abs(a - b) for a, b in zip(values, clipped, strict=True)]) / denom
