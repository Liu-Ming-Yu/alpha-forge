"""Sample-free intraday alpha backfill input construction."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.services.research_service.sampling.samples import (
    SupervisedAlphaSample,
    research_as_of_dates,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping

    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.core.domain.research import FeatureVector


async def _sample_free_intraday_samples(
    *,
    feature_repo: FeatureRepository,
    contracts: Mapping[uuid.UUID, Mapping[str, object]],
    start: datetime,
    end: datetime,
    date_policy: str,
    context_feature_set_version: str = "",
) -> tuple[tuple[SupervisedAlphaSample, ...], dict[str, object]]:
    """Build current-evidence decision samples without forward-return labels."""
    instrument_ids = tuple(contracts)
    as_of_dates = research_as_of_dates(start, end, date_policy=date_policy)
    samples: list[SupervisedAlphaSample] = []
    context_vectors_used = 0
    context_vectors_missing = 0
    for as_of in as_of_dates:
        exact_context: dict[uuid.UUID, FeatureVector] = {}
        if context_feature_set_version:
            vectors = await feature_repo.get_vectors(
                list(instrument_ids),
                context_feature_set_version,
                as_of,
            )
            exact_context = {
                vector.instrument_id: vector
                for vector in vectors
                if _ensure_utc(vector.as_of) == _ensure_utc(as_of)
                and _ensure_utc(vector.available_at or vector.as_of) <= _ensure_utc(as_of)
            }
        for instrument_id in instrument_ids:
            features: dict[str, float] = {}
            vector = exact_context.get(instrument_id)
            if vector is not None:
                features = dict(vector.features)
                context_vectors_used += 1
            elif context_feature_set_version:
                context_vectors_missing += 1
            samples.append(
                SupervisedAlphaSample(
                    as_of=_ensure_utc(as_of),
                    instrument_id=instrument_id,
                    features=features,
                    forward_return=0.0,
                    metadata=(("label_mode", "sample_free_current_evidence"),),
                )
            )
    return tuple(samples), {
        "mode": "sample_free",
        "start": _ensure_utc(start).isoformat(),
        "end": _ensure_utc(end).isoformat(),
        "date_policy": date_policy,
        "as_of_dates": len(as_of_dates),
        "samples": len(samples),
        "context_feature_set_version": context_feature_set_version,
        "context_vectors_used": context_vectors_used,
        "context_vectors_missing": context_vectors_missing,
    }


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
