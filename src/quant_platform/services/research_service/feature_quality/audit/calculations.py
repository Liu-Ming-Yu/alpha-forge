"""Pure calculations used by feature-audit gates."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import FeatureExpectedSign
from quant_platform.services.research_service.reports.statistics import (
    mean,
    pearson,
    spearman_ic,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def feature_rows(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
) -> tuple[SupervisedAlphaSample, ...]:
    return tuple(
        row
        for row in samples
        if feature_name in row.features and math.isfinite(float(row.features[feature_name]))
    )


def values_by_day(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
) -> dict[datetime, list[float]]:
    by_day: dict[datetime, list[float]] = {}
    for row in samples:
        value = row.features.get(feature_name)
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            by_day.setdefault(row.as_of, []).append(numeric)
    return by_day


def daily_ic(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    *,
    sign: float = 1.0,
) -> list[float]:
    by_day: dict[datetime, list[SupervisedAlphaSample]] = {}
    for row in samples:
        by_day.setdefault(row.as_of, []).append(row)
    out: list[float] = []
    for _, rows in sorted(by_day.items()):
        scores: list[float] = []
        labels: list[float] = []
        for row in rows:
            if feature_name not in row.features:
                continue
            scores.append(float(row.features[feature_name]) * sign)
            labels.append(row.forward_return)
        if len(scores) >= 2:
            out.append(spearman_ic(scores, labels))
    return out


def lagged_feature_ic(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    *,
    sign: float,
) -> float:
    by_day: dict[datetime, list[SupervisedAlphaSample]] = {}
    for row in samples:
        by_day.setdefault(row.as_of, []).append(row)
    prior: dict[uuid.UUID, float] = {}
    ics: list[float] = []
    for _, rows in sorted(by_day.items()):
        scores: list[float] = []
        labels: list[float] = []
        for row in rows:
            if row.instrument_id in prior:
                scores.append(prior[row.instrument_id] * sign)
                labels.append(row.forward_return)
        if len(scores) >= 2:
            ics.append(spearman_ic(scores, labels))
        for row in rows:
            if feature_name in row.features:
                prior[row.instrument_id] = float(row.features[feature_name])
    return mean(ics)


def permuted_ic(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    seed: int,
) -> float:
    rng = random.Random(seed)
    by_day: dict[datetime, list[SupervisedAlphaSample]] = {}
    for row in samples:
        by_day.setdefault(row.as_of, []).append(row)
    ics: list[float] = []
    for _, rows in sorted(by_day.items()):
        scores = [float(row.features[feature_name]) for row in rows if feature_name in row.features]
        labels = [row.forward_return for row in rows if feature_name in row.features]
        if len(scores) < 3:
            continue
        rng.shuffle(scores)
        ics.append(spearman_ic(scores, labels))
    return mean(ics)


def availability_violations(samples: Sequence[SupervisedAlphaSample]) -> int:
    count = 0
    for row in samples:
        metadata = row.metadata_dict()
        for key in ("available_at", "source_available_at"):
            raw = metadata.get(key)
            if not raw:
                continue
            try:
                available = datetime.fromisoformat(raw)
            except ValueError:
                count += 1
                continue
            if available.tzinfo is None:
                available = available.replace(tzinfo=UTC)
            if available > row.as_of:
                count += 1
    return count


def feature_weighted_returns(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    *,
    sign: float,
    slippage_bps: float,
) -> tuple[list[float], list[float]]:
    by_day: dict[datetime, list[SupervisedAlphaSample]] = {}
    for row in samples:
        by_day.setdefault(row.as_of, []).append(row)
    returns: list[float] = []
    turnovers: list[float] = []
    prior_weights: dict[uuid.UUID, float] = {}
    cost_per_turnover = slippage_bps / 10_000.0
    for _, rows in sorted(by_day.items()):
        raw_scores = {
            row.instrument_id: float(row.features[feature_name]) * sign
            for row in rows
            if feature_name in row.features
        }
        denom = sum(abs(value) for value in raw_scores.values())
        weights = (
            {iid: value / denom for iid, value in raw_scores.items()}
            if denom > 0
            else {iid: 0.0 for iid in raw_scores}
        )
        keys = set(weights) | set(prior_weights)
        turnover = sum(abs(weights.get(key, 0.0) - prior_weights.get(key, 0.0)) for key in keys)
        forward = {
            row.instrument_id: row.forward_return for row in rows if row.instrument_id in weights
        }
        pnl = sum(weights[iid] * forward.get(iid, 0.0) for iid in weights)
        returns.append(pnl - turnover * cost_per_turnover)
        turnovers.append(turnover)
        prior_weights = weights
    return returns, turnovers


def feature_score_by_day(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    sign: float,
) -> dict[datetime, list[tuple[uuid.UUID, float, float]]]:
    out: dict[datetime, list[tuple[uuid.UUID, float, float]]] = {}
    for row in samples:
        if feature_name not in row.features:
            continue
        out.setdefault(row.as_of, []).append(
            (row.instrument_id, float(row.features[feature_name]) * sign, row.forward_return)
        )
    return out


def baseline_score_by_day(
    samples: Sequence[SupervisedAlphaSample],
    feature_names: Sequence[str],
) -> dict[datetime, list[tuple[uuid.UUID, float, float]]]:
    if not feature_names:
        return {}
    out: dict[datetime, list[tuple[uuid.UUID, float, float]]] = {}
    for row in samples:
        values = [float(row.features[name]) for name in feature_names if name in row.features]
        if not values:
            continue
        out.setdefault(row.as_of, []).append((row.instrument_id, mean(values), row.forward_return))
    return out


def daily_score_ic(
    scores_by_day: Mapping[datetime, Sequence[tuple[uuid.UUID, float, float]]],
) -> list[float]:
    out: list[float] = []
    for _, rows in sorted(scores_by_day.items()):
        if len(rows) < 2:
            continue
        scores = [item[1] for item in rows]
        labels = [item[2] for item in rows]
        out.append(spearman_ic(scores, labels))
    return out


def combine_scores(
    candidate: Mapping[datetime, Sequence[tuple[uuid.UUID, float, float]]],
    baseline: Mapping[datetime, Sequence[tuple[uuid.UUID, float, float]]],
) -> dict[datetime, list[tuple[uuid.UUID, float, float]]]:
    out: dict[datetime, list[tuple[uuid.UUID, float, float]]] = {}
    for day, cand_rows in candidate.items():
        base_rows = {iid: score for iid, score, _ in baseline.get(day, ())}
        rows: list[tuple[uuid.UUID, float, float]] = []
        for iid, score, label in cand_rows:
            rows.append((iid, score + base_rows.get(iid, 0.0), label))
        out[day] = rows
    return out


def max_baseline_corr(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    baseline_features: Sequence[str],
) -> float:
    if not baseline_features:
        return 0.0
    candidate = [float(row.features.get(feature_name, 0.0)) for row in samples]
    corrs = []
    for name in baseline_features:
        values = [float(row.features.get(name, 0.0)) for row in samples]
        corrs.append(abs(pearson(candidate, values)))
    return max(corrs, default=0.0)


def sign_multiplier(expected: FeatureExpectedSign) -> float:
    return -1.0 if expected == FeatureExpectedSign.NEGATIVE else 1.0
