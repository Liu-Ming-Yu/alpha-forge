"""Metrics for feature failure attribution diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import exchange_calendars as xcals

from quant_platform.services.research_service.feature_quality.audit.calculations import (
    availability_violations,
)
from quant_platform.services.research_service.reports.statistics import (
    mean,
    negative_streak,
    pearson,
    spearman_ic,
    std_sample,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def feature_names(samples: Sequence[SupervisedAlphaSample]) -> tuple[str, ...]:
    """Return sorted non-reserved feature names in a sample set."""
    return tuple(
        sorted(
            {
                str(name)
                for sample in samples
                for name in sample.features
                if not str(name).startswith("_")
            }
        )
    )


def ic_summary(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    sign: float,
) -> dict[str, float]:
    """Summarise daily cross-sectional IC for one feature."""
    values = [_float_value(row.get("ic")) for row in daily_ic_rows(samples, feature_name, sign)]
    value_mean = mean(values)
    ic_std = std_sample(values)
    return {
        "ic_mean": value_mean,
        "icir": value_mean / ic_std if ic_std > 0 else (999.0 if value_mean > 0 else 0.0),
        "ic_negative_streak": float(negative_streak(values)),
        "ic_observations": float(len(values)),
    }


def daily_ic_rows(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    sign: float,
) -> list[dict[str, object]]:
    """Return date-stamped daily IC rows for one feature."""
    grouped = samples_by_day(samples)
    rows: list[dict[str, object]] = []
    for as_of, day_samples in sorted(grouped.items()):
        scores, labels = feature_scores_and_labels(day_samples, feature_name, sign)
        if len(scores) >= 2:
            rows.append(
                {
                    "as_of": as_of.astimezone(UTC).isoformat(),
                    "ic": spearman_ic(scores, labels),
                }
            )
    return rows


def null_baseline(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    sign: float,
    *,
    seed: int,
    count: int,
) -> dict[str, float]:
    """Build a deterministic within-date permutation null baseline."""
    actual = ic_summary(samples, feature_name, sign)["ic_mean"]
    grouped = samples_by_day(samples)
    means: list[float] = []
    for iteration in range(max(1, count)):
        ics: list[float] = []
        for day_index, day_samples in enumerate(grouped.values()):
            scores, labels = feature_scores_and_labels(day_samples, feature_name, sign)
            if len(scores) < 2:
                continue
            shuffled = deterministic_shuffle(labels, seed + iteration * 997 + day_index)
            ics.append(spearman_ic(scores, shuffled))
        means.append(mean(ics))
    means.sort()
    exceedances = sum(1 for value in means if value >= actual)
    return {
        "actual_ic_mean": actual,
        "null_mean": mean(means),
        "null_p05": quantile(means, 0.05),
        "null_p95": quantile(means, 0.95),
        "one_sided_p_value": (exceedances + 1.0) / (len(means) + 1.0),
        "permutations": float(len(means)),
    }


def correlation_matrix(
    samples: Sequence[SupervisedAlphaSample],
    names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute feature-value Pearson correlations across sample rows."""
    matrix: dict[str, dict[str, float]] = {}
    for left in names:
        matrix[left] = {}
        for right in names:
            pairs = [
                (float(sample.features[left]), float(sample.features[right]))
                for sample in samples
                if left in sample.features and right in sample.features
            ]
            matrix[left][right] = pearson([a for a, _ in pairs], [b for _, b in pairs])
    return matrix


def correlation_clusters(
    matrix: Mapping[str, Mapping[str, float]],
    threshold: float,
) -> list[dict[str, object]]:
    """Return connected components under an absolute-correlation threshold."""
    remaining = set(matrix)
    clusters: list[dict[str, object]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        members: set[str] = set()
        while stack:
            item = stack.pop()
            if item in members:
                continue
            members.add(item)
            stack.extend(
                peer
                for peer, corr in matrix.get(item, {}).items()
                if peer not in members and abs(float(corr)) >= threshold
            )
        remaining -= members
        if len(members) > 1:
            clusters.append({"features": sorted(members), "size": len(members)})
    return clusters


def data_validation(
    *,
    official_samples: Sequence[SupervisedAlphaSample],
    samples_by_horizon: Mapping[int, Sequence[SupervisedAlphaSample]],
    sample_builds_by_horizon: Mapping[int, Mapping[str, object]],
    date_policy: str,
    nested_object_store_present: bool,
) -> dict[str, object]:
    """Summarise point-in-time and calendar validation for attribution inputs."""
    non_sessions = non_nyse_session_dates(official_samples)
    non_session_set = set(non_sessions)
    return {
        "date_policy": date_policy,
        "nyse_session_only": not non_sessions,
        "non_nyse_session_dates": non_sessions,
        "weekend_or_holiday_feature_vector_count": sum(
            1 for sample in official_samples if sample.as_of.date().isoformat() in non_session_set
        ),
        "available_at_violations": availability_violations(official_samples),
        "sample_counts_by_horizon": {
            str(horizon): len(samples) for horizon, samples in sorted(samples_by_horizon.items())
        },
        "sample_builds_by_horizon": {
            str(horizon): dict(payload)
            for horizon, payload in sorted(sample_builds_by_horizon.items())
        },
        "nested_data_parquet_present": nested_object_store_present,
    }


def worst_negative_streak_window(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return the longest negative-IC streak window."""
    best_start = ""
    best_end = ""
    best_len = 0
    current_start = ""
    current_len = 0
    for row in rows:
        as_of = str(row["as_of"])
        if _float_value(row.get("ic")) < 0:
            current_start = current_start or as_of
            current_len += 1
            if current_len > best_len:
                best_start = current_start
                best_end = as_of
                best_len = current_len
            continue
        current_start = ""
        current_len = 0
    return {"length": best_len, "start": best_start or None, "end": best_end or None}


def monthly_ic_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate daily IC rows by calendar month."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["as_of"])[:7]].append(_float_value(row.get("ic")))
    return [
        {
            "month": month,
            "ic_mean": mean(values),
            "negative_days": sum(1 for value in values if value < 0),
            "observations": len(values),
        }
        for month, values in sorted(grouped.items())
    ]


def samples_by_day(
    samples: Sequence[SupervisedAlphaSample],
) -> dict[datetime, list[SupervisedAlphaSample]]:
    grouped: dict[datetime, list[SupervisedAlphaSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.as_of].append(sample)
    return grouped


def feature_scores_and_labels(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    sign: float,
) -> tuple[list[float], list[float]]:
    scores: list[float] = []
    labels: list[float] = []
    for sample in samples:
        value = sample.features.get(feature_name)
        if value is None or not math.isfinite(float(value)):
            continue
        scores.append(float(value) * sign)
        labels.append(float(sample.forward_return))
    return scores, labels


def non_nyse_session_dates(samples: Sequence[SupervisedAlphaSample]) -> list[str]:
    """Return sample dates that are not XNYS sessions."""
    dates = sorted({sample.as_of.astimezone(UTC).date().isoformat() for sample in samples})
    if not dates:
        return []
    calendar = xcals.get_calendar("XNYS")
    sessions = {
        session.date().isoformat() for session in calendar.sessions_in_range(dates[0], dates[-1])
    }
    return [date for date in dates if date not in sessions]


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    index = int(q * (len(values) - 1))
    return values[max(0, min(len(values) - 1, index))]


def deterministic_shuffle(values: Sequence[float], seed: int) -> list[float]:
    items = list(values)
    state = seed & 0x7FFFFFFF
    for index in range(len(items) - 1, 0, -1):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        swap_index = state % (index + 1)
        items[index], items[swap_index] = items[swap_index], items[index]
    return items


def _float_value(raw: object, default: float = 0.0) -> float:
    if not isinstance(raw, int | float | str):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError, OverflowError):
        return default


__all__ = [
    "correlation_clusters",
    "correlation_matrix",
    "daily_ic_rows",
    "data_validation",
    "feature_names",
    "ic_summary",
    "monthly_ic_rows",
    "null_baseline",
    "worst_negative_streak_window",
]
