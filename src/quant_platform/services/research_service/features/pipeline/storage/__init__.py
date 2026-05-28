"""Feature vector storage helpers shared by feature-pipeline entrypoints."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.research import FeatureResult, FeatureVector
from quant_platform.telemetry.feature_metrics import emit_feature_distribution_metrics

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.services.research_service.features.cross_section.cross_section import (
        FeatureBundle,
    )

log = structlog.get_logger(__name__)

# Reserved feature keys are prefixed with an underscore so alpha-feature
# consumers can skip them by convention when iterating on feature names.
VOL_FORECAST_KEY = "_vol_forecast_annualized"
_VOL_FORECAST_MIN = 1e-6
_VOL_FORECAST_MAX = 10.0


def _merge_alpha_and_vol(
    alpha_features: Mapping[uuid.UUID, Mapping[str, float]],
    vol_forecasts: Mapping[uuid.UUID, float],
) -> dict[uuid.UUID, dict[str, float]]:
    """Merge alpha features with the reserved vol-forecast key per instrument."""
    from quant_platform.core.exceptions import FeatureValidationError

    merged: dict[uuid.UUID, dict[str, float]] = {}
    for instrument_id, alpha in alpha_features.items():
        row: dict[str, float] = dict(alpha)
        forecast = vol_forecasts.get(instrument_id)
        if forecast is not None and forecast > 0.0:
            if not (_VOL_FORECAST_MIN < forecast <= _VOL_FORECAST_MAX):
                raise FeatureValidationError(
                    f"vol_forecast out of bounds for instrument {instrument_id}: "
                    f"{forecast:.6f} not in [{_VOL_FORECAST_MIN}, {_VOL_FORECAST_MAX}]"
                )
            row[VOL_FORECAST_KEY] = float(forecast)
        merged[instrument_id] = row
    return merged


def extract_vol_forecasts(
    feature_data: Mapping[uuid.UUID, Mapping[str, float]],
) -> dict[uuid.UUID, float]:
    """Pull positive reserved vol-forecast values out of feature data."""
    out: dict[uuid.UUID, float] = {}
    for instrument_id, row in feature_data.items():
        value = row.get(VOL_FORECAST_KEY)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0:
            out[instrument_id] = numeric
    return out


async def compute_and_store_feature_bundle(
    bundle: FeatureBundle,
    *,
    repo: FeatureRepository,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    feature_set_version: str,
    artifact_uri: str,
    available_at: datetime | None,
) -> dict[uuid.UUID, dict[str, float]]:
    merged = _merge_alpha_and_vol(bundle.alpha_features, bundle.vol_forecasts)
    await _store_feature_vectors(
        merged=merged,
        repo=repo,
        strategy_run_id=strategy_run_id,
        as_of=as_of,
        feature_set_version=feature_set_version,
        artifact_uri=artifact_uri,
        available_at=available_at,
    )
    return merged


def feature_result_from_bundle(
    bundle: FeatureBundle,
    *,
    feature_set_version: str,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    artifact_uri: str = "",
    available_at: datetime | None = None,
) -> FeatureResult:
    """Convert a :class:`FeatureBundle` into a pure typed :class:`FeatureResult`.

    This is the persistence-free counterpart of :func:`compute_and_store_feature_bundle`:
    it merges alpha features with the reserved vol-forecast key and packages the
    per-instrument rows as immutable ``FeatureVector`` values without touching a
    repository. ``FeatureFamilyRegistry`` computers use it; the caller persists.
    """
    merged = _merge_alpha_and_vol(bundle.alpha_features, bundle.vol_forecasts)
    vectors = tuple(
        FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=instrument_id,
            as_of=as_of,
            feature_set_version=feature_set_version,
            features=features,
            strategy_run_id=strategy_run_id,
            artifact_uri=artifact_uri,
            available_at=available_at or as_of,
        )
        for instrument_id, features in merged.items()
    )
    return FeatureResult(
        feature_set_version=feature_set_version,
        vectors=vectors,
        diagnostics={"instruments": len(vectors)},
        passed=True,
    )


async def _store_feature_vectors(
    *,
    merged: dict[uuid.UUID, dict[str, float]],
    repo: FeatureRepository,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    feature_set_version: str,
    artifact_uri: str,
    available_at: datetime | None,
) -> None:
    emit_feature_distribution_metrics(merged, feature_set_version)

    stored = 0
    for instrument_id, features in merged.items():
        await repo.store_vector(
            FeatureVector(
                vector_id=uuid.uuid4(),
                instrument_id=instrument_id,
                as_of=as_of,
                feature_set_version=feature_set_version,
                features=features,
                strategy_run_id=strategy_run_id,
                artifact_uri=artifact_uri,
                available_at=available_at or as_of,
            )
        )
        stored += 1

    log.info(
        "feature_pipeline.stored",
        instruments=stored,
        version=feature_set_version,
        vol_forecasts=sum(1 for row in merged.values() if VOL_FORECAST_KEY in row),
        as_of=str(as_of),
        artifact_uri=artifact_uri or "(none)",
    )


__all__ = [
    "VOL_FORECAST_KEY",
    "compute_and_store_feature_bundle",
    "extract_vol_forecasts",
    "feature_result_from_bundle",
]
