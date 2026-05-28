"""Feature attachment for text candidate screens."""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.screening.common import ensure_utc
from quant_platform.services.research_service.text.candidates.catalog import (
    TextCandidateSpec,
    build_text_aggregate_features,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from quant_platform.core.domain.research import FeatureVector
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def _with_screened_features(
    *,
    samples: Sequence[SupervisedAlphaSample],
    text_vectors: Sequence[FeatureVector],
    candidates: Sequence[TextCandidateSpec],
    lookback_days: int,
) -> tuple[SupervisedAlphaSample, ...]:
    vectors_by_instrument = _vectors_by_instrument(text_vectors)
    aggregate_rows: list[
        tuple[SupervisedAlphaSample, dict[str, float], float | None, datetime]
    ] = []
    samples_by_as_of: dict[datetime, int] = defaultdict(int)
    active_text_by_as_of: dict[datetime, int] = defaultdict(int)
    for sample in samples:
        as_of_key = ensure_utc(sample.as_of)
        text_features, decay = build_text_aggregate_features(
            vectors_by_instrument.get(sample.instrument_id, ()),
            sample.as_of,
            lookback_days=lookback_days,
        )
        samples_by_as_of[as_of_key] += 1
        if decay is not None:
            active_text_by_as_of[as_of_key] += 1
        aggregate_rows.append((sample, text_features, decay, as_of_key))

    coverage_by_as_of = {
        as_of: (active_text_by_as_of[as_of] / total if total else 0.0)
        for as_of, total in samples_by_as_of.items()
    }

    screened: list[SupervisedAlphaSample] = []
    for sample, text_features, decay, as_of_key in aggregate_rows:
        text_context = dict(text_features)
        text_context["text_cross_section_coverage_21d"] = coverage_by_as_of.get(
            as_of_key,
            0.0,
        )
        features = dict(sample.features)
        for candidate in candidates:
            value = 0.0
            if decay is not None:
                value = candidate.formula(text_context, sample.features, decay)
                if not math.isfinite(value):
                    value = 0.0
            features[candidate.name] = value
        screened.append(dataclasses.replace(sample, features=features))
    return tuple(screened)


def _vectors_by_instrument(
    text_vectors: Sequence[FeatureVector],
) -> dict[object, tuple[FeatureVector, ...]]:
    grouped: dict[object, list[FeatureVector]] = defaultdict(list)
    for vector in text_vectors:
        grouped[vector.instrument_id].append(vector)
    return {
        instrument_id: tuple(sorted(rows, key=lambda row: ensure_utc(row.as_of)))
        for instrument_id, rows in grouped.items()
    }
