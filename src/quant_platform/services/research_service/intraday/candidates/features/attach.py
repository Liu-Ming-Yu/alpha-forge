"""Attach intraday candidate features to research samples."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.screening.common import ensure_utc
from quant_platform.services.research_service.intraday.candidates.features.sessions import (
    _sessions_before,
    _sessions_by_instrument,
)
from quant_platform.services.research_service.intraday.candidates.features.types import (
    INTRADAY_ALPHA_SCALE,
    IntradayCandidateFeatureRow,
    IntradayCandidateFeatureSpec,
    IntradayFeatureContext,
    IntradaySessionMetrics,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def attach_intraday_candidate_features(
    *,
    samples: Sequence[SupervisedAlphaSample],
    intraday_bars: Sequence[MarketBar],
    candidates: Sequence[IntradayCandidateFeatureSpec],
) -> tuple[SupervisedAlphaSample, ...]:
    sessions = _sessions_by_instrument(intraday_bars)
    screened: list[SupervisedAlphaSample] = []
    for sample in samples:
        history = _sessions_before(sessions.get(sample.instrument_id, ()), sample.as_of)
        metrics = history[-1] if history else None
        features = dict(sample.features)
        for candidate in candidates:
            features[candidate.name] = _candidate_value(
                candidate,
                metrics,
                sample.as_of,
                history=history,
                sample_features=sample.features,
            )
        screened.append(dataclasses.replace(sample, features=features))
    return tuple(screened)


def build_intraday_candidate_feature_rows(
    *,
    samples: Sequence[SupervisedAlphaSample],
    intraday_bars: Sequence[MarketBar],
    candidates: Sequence[IntradayCandidateFeatureSpec],
) -> tuple[IntradayCandidateFeatureRow, ...]:
    sessions = _sessions_by_instrument(intraday_bars)
    rows: list[IntradayCandidateFeatureRow] = []
    for sample in samples:
        as_of = ensure_utc(sample.as_of)
        history = _sessions_before(sessions.get(sample.instrument_id, ()), as_of)
        metrics = history[-1] if history else None
        if metrics is None or metrics.end_at > as_of:
            continue
        rows.append(
            IntradayCandidateFeatureRow(
                as_of=as_of,
                instrument_id=sample.instrument_id,
                available_at=metrics.end_at,
                features={
                    candidate.name: _candidate_value(
                        candidate,
                        metrics,
                        as_of,
                        history=history,
                        sample_features=sample.features,
                    )
                    for candidate in candidates
                },
            )
        )
    return tuple(rows)


def _candidate_value(
    spec: IntradayCandidateFeatureSpec,
    metrics: IntradaySessionMetrics | None,
    as_of: datetime,
    *,
    history: Sequence[IntradaySessionMetrics] = (),
    sample_features: Mapping[str, float] | None = None,
) -> float:
    if metrics is None:
        return 0.0
    age_days = (ensure_utc(as_of) - metrics.end_at).total_seconds() / 86400.0
    if age_days < 0 or age_days > spec.lookback_days:
        return 0.0
    decay = max(0.0, 1.0 - (age_days / max(float(spec.lookback_days), 1.0)))
    return INTRADAY_ALPHA_SCALE * spec.formula(
        IntradayFeatureContext(
            metrics=metrics,
            decay=decay,
            history=tuple(history),
            as_of=ensure_utc(as_of),
            sample_features=sample_features,
        )
    )
