"""Shared internals for text feature bundle builders."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.screening.common import (
    ensure_utc as _ensure_utc,
)
from quant_platform.services.research_service.features.cross_section.cross_section import (
    FeatureBundle,
    build_feature_bundle,
)
from quant_platform.services.research_service.text.candidates.catalog import (
    build_text_aggregate_features,
    text_candidate_specs_by_name,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Mapping, Sequence
    from datetime import datetime

    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.core.domain.research import FeatureVector


async def _build_text_feature_bundle(
    *,
    bar_data: Mapping[uuid.UUID, Sequence[MarketBar]],
    text_feature_repo: FeatureRepository,
    as_of: datetime,
    text_feature_set_version: str,
    lookback_days: int,
    decayed_features: Callable[[FeatureVector | None, datetime, int], dict[str, float]],
) -> FeatureBundle:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    as_of_utc = _ensure_utc(as_of)
    close_history = {
        instrument_id: [float(bar.close) for bar in bars]
        for instrument_id, bars in bar_data.items()
    }
    classical = build_feature_bundle(close_history)
    instrument_ids = tuple(bar_data)
    text_vectors = await text_feature_repo.get_vectors(
        list(instrument_ids),
        text_feature_set_version,
        as_of_utc,
    )
    by_instrument = {vector.instrument_id: vector for vector in text_vectors}

    alpha_features: dict[uuid.UUID, dict[str, float]] = {}
    for instrument_id in instrument_ids:
        row = dict(classical.alpha_features.get(instrument_id, {}))
        row.update(decayed_features(by_instrument.get(instrument_id), as_of_utc, lookback_days))
        alpha_features[instrument_id] = row

    return FeatureBundle(
        alpha_features=alpha_features,
        vol_forecasts=dict(classical.vol_forecasts),
    )


async def _build_text_candidate_feature_bundle(
    *,
    bar_data: Mapping[uuid.UUID, Sequence[MarketBar]],
    text_feature_repo: FeatureRepository,
    as_of: datetime,
    text_feature_set_version: str,
    lookback_days: int,
    feature_names: Sequence[str],
    candidate_set: str,
) -> FeatureBundle:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    as_of_utc = _ensure_utc(as_of)
    close_history = {
        instrument_id: [float(bar.close) for bar in bars]
        for instrument_id, bars in bar_data.items()
    }
    classical = build_feature_bundle(close_history)
    instrument_ids = tuple(bar_data)
    text_vectors = await _get_visible_text_vector_history(
        text_feature_repo,
        list(instrument_ids),
        text_feature_set_version,
        as_of_utc,
        lookback_days=lookback_days,
    )
    vectors_by_instrument: dict[uuid.UUID, list[FeatureVector]] = defaultdict(list)
    for vector in text_vectors:
        vectors_by_instrument[vector.instrument_id].append(vector)
    specs = text_candidate_specs_by_name(candidate_set)

    aggregates_by_instrument: dict[uuid.UUID, tuple[dict[str, float], float | None]] = {}
    active_text_count = 0
    for instrument_id in instrument_ids:
        aggregate, decay = build_text_aggregate_features(
            vectors_by_instrument.get(instrument_id, ()),
            as_of_utc,
            lookback_days=lookback_days,
        )
        if decay is not None:
            active_text_count += 1
        aggregates_by_instrument[instrument_id] = (aggregate, decay)
    text_cross_section_coverage = active_text_count / len(instrument_ids) if instrument_ids else 0.0

    alpha_features: dict[uuid.UUID, dict[str, float]] = {}
    for instrument_id in instrument_ids:
        row = dict(classical.alpha_features.get(instrument_id, {}))
        aggregate, decay = aggregates_by_instrument[instrument_id]
        text_context = dict(aggregate)
        text_context["text_cross_section_coverage_21d"] = text_cross_section_coverage
        for feature_name in feature_names:
            value = 0.0
            spec = specs.get(feature_name)
            if spec is not None and decay is not None:
                value = spec.formula(text_context, row, decay)
                if not math.isfinite(value):
                    value = 0.0
            row[feature_name] = value
        alpha_features[instrument_id] = row

    return FeatureBundle(
        alpha_features=alpha_features,
        vol_forecasts=dict(classical.vol_forecasts),
    )


async def _get_visible_text_vector_history(
    repo: FeatureRepository,
    instrument_ids: list[uuid.UUID],
    feature_set_version: str,
    as_of: datetime,
    *,
    lookback_days: int,
) -> tuple[FeatureVector, ...]:
    """Recover daily visible text vectors without widening the repository protocol."""
    start = as_of - timedelta(days=lookback_days)
    history_loader = getattr(repo, "get_vector_history", None)
    if callable(history_loader):
        rows = await history_loader(
            instrument_ids,
            feature_set_version,
            start,
            as_of,
        )
        return tuple(
            sorted(
                rows,
                key=lambda row: (_ensure_utc(row.as_of), row.instrument_id),
            )
        )

    by_key: dict[tuple[uuid.UUID, str, datetime], FeatureVector] = {}
    for offset in range(lookback_days + 1):
        anchor = start + timedelta(days=offset)
        for vector in await repo.get_vectors(instrument_ids, feature_set_version, anchor):
            key = (vector.instrument_id, vector.feature_set_version, _ensure_utc(vector.as_of))
            by_key[key] = vector
    return tuple(
        sorted(
            by_key.values(),
            key=lambda row: (_ensure_utc(row.as_of), row.instrument_id),
        )
    )
