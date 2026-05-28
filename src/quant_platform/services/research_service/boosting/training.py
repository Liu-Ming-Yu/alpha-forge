"""Training-data helpers for XGBoost ranker research models."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from quant_platform.services.research_service.reports.statistics import spearman_ic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.services.research_service.boosting.artifacts import BoostingSample


def infer_feature_names(samples: Sequence[BoostingSample]) -> list[str]:
    names: set[str] = set()
    for sample in samples:
        for name in sample.features:
            if not name.startswith("_"):
                names.add(name)
    if not names:
        raise ValueError("boosting training requires at least one non-reserved feature")
    return sorted(names)


def build_matrix(
    xgb: object,
    samples: Sequence[BoostingSample],
    feature_names: Sequence[str],
) -> object:
    if not samples:
        raise ValueError("cannot build XGBoost matrix from empty sample set")
    rows: list[list[float]] = []
    labels: list[float] = []
    groups: list[int] = []
    grouped = _group_samples_by_as_of(samples)

    for group in grouped:
        relevance = integer_relevance_labels(group)
        groups.append(len(group))
        for sample, label in zip(group, relevance, strict=True):
            row = []
            for name in feature_names:
                value = float(sample.features.get(name, 0.0))
                if not math.isfinite(value):
                    raise ValueError(f"boosting sample feature {name!r} is not finite")
                row.append(value)
            rows.append(row)
            labels.append(float(label))
    xgb_any = cast("Any", xgb)
    dmat = xgb_any.DMatrix(
        np.asarray(rows, dtype=float),
        label=np.asarray(labels, dtype=float),
        feature_names=list(feature_names),
    )
    dmat.set_group(np.asarray(groups, dtype=np.uint32))
    return dmat


def integer_relevance_labels(samples: Sequence[BoostingSample]) -> list[int]:
    """Map raw forward returns to non-negative integer relevance labels per date."""
    unique_returns = sorted({float(sample.forward_return) for sample in samples})
    relevance_by_return = {value: idx for idx, value in enumerate(unique_returns)}
    return [relevance_by_return[float(sample.forward_return)] for sample in samples]


def split_samples(
    samples: Sequence[BoostingSample],
    *,
    validation_fraction: float,
    purge_days: int,
) -> tuple[list[BoostingSample], list[BoostingSample], datetime]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    sorted_all = sorted(samples, key=lambda sample: (sample.as_of, str(sample.instrument_id)))
    timestamps = sorted({sample.as_of for sample in sorted_all})
    if len(timestamps) < 2:
        raise ValueError("boosting training requires at least two as_of groups")
    validation_groups = max(1, int(math.ceil(len(timestamps) * validation_fraction)))
    validation_start = timestamps[-validation_groups]
    train_cutoff = validation_start - timedelta(days=purge_days)
    train = [sample for sample in sorted_all if sample.as_of < train_cutoff]
    valid = [sample for sample in sorted_all if sample.as_of >= validation_start]
    if not train or not valid:
        raise ValueError("time split produced empty train or validation sample set")
    return train, valid, validation_start


def validation_ic(
    xgb: object,
    booster: object,
    samples: Sequence[BoostingSample],
    dmat: object,
) -> float:
    booster_any = cast("Any", booster)
    predictions = booster_any.predict(dmat)
    sorted_samples = sorted(samples, key=lambda sample: (sample.as_of, str(sample.instrument_id)))
    by_group: dict[datetime, tuple[list[float], list[float]]] = {}
    for sample, pred in zip(sorted_samples, predictions, strict=True):
        preds, labels = by_group.setdefault(sample.as_of, ([], []))
        preds.append(float(pred))
        labels.append(sample.forward_return)
    ics = [spearman_ic(preds, labels) for preds, labels in by_group.values() if len(preds) >= 2]
    del xgb
    return float(sum(ics) / len(ics)) if ics else 0.0


def _group_samples_by_as_of(samples: Sequence[BoostingSample]) -> list[list[BoostingSample]]:
    sorted_samples = sorted(samples, key=lambda sample: (sample.as_of, str(sample.instrument_id)))
    grouped: list[list[BoostingSample]] = []
    current_group: list[BoostingSample] = []
    current_ts: datetime | None = None
    for sample in sorted_samples:
        if current_ts is None or sample.as_of != current_ts:
            if current_group:
                grouped.append(current_group)
            current_group = []
            current_ts = sample.as_of
        current_group.append(sample)
    if current_group:
        grouped.append(current_group)
    return grouped
