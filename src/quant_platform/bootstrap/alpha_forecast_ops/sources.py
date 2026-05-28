"""Per-source forecast builders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.bootstrap.alpha_forecast_ops.payloads import (
    _ensure_utc,
    _feature_coverage,
    _linear_score,
    _prediction_result,
    symbol_for,
)
from quant_platform.bootstrap.alpha_forecast_ops.policies import _linear_source_policy

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.core.domain.production import PredictionResult
    from quant_platform.core.domain.research import FeatureVector, StrategyRun
    from quant_platform.services.research_service.boosting.model import XGBoostRankSignalModel


async def _linear_source_predictions(
    settings: PlatformSettings,
    *,
    source: str,
    feature_repo: FeatureRepository,
    instrument_ids: Sequence[uuid.UUID],
    contracts: Mapping[uuid.UUID, Mapping[str, object]],
    as_of: datetime,
    horizon: str,
    strategy_run: StrategyRun,
) -> tuple[list[PredictionResult], dict[str, object]]:
    feature_set_version, model_version, weights = _linear_source_policy(settings, source)
    feature_names = tuple(weights)
    schema_hash = ordered_feature_schema_hash(feature_names)
    vectors = await feature_repo.get_vectors(list(instrument_ids), feature_set_version, as_of)
    exact_vectors = {
        vector.instrument_id: vector
        for vector in vectors
        if _ensure_utc(vector.as_of) == as_of
        and _ensure_utc(vector.available_at or vector.as_of) <= as_of
    }
    blockers = _missing_vector_blockers(
        source,
        instrument_ids=instrument_ids,
        vectors=exact_vectors,
        contracts=contracts,
    )
    records: list[PredictionResult] = []
    for instrument_id in instrument_ids:
        vector = exact_vectors.get(instrument_id)
        if vector is None:
            continue
        coverage = _feature_coverage(vector.features, feature_names)
        missing_features = [name for name in feature_names if name not in vector.features]
        if missing_features:
            blockers.append(
                f"{source} {symbol_for(contracts, instrument_id)} missing features: "
                f"{', '.join(missing_features)}"
            )
            if coverage <= 0.0:
                continue
        score = _linear_score(vector.features, weights)
        records.append(
            _prediction_result(
                source=source,
                model_version=model_version,
                instrument_id=instrument_id,
                strategy_run_id=strategy_run.run_id,
                as_of=as_of,
                horizon=horizon,
                expected_return=score,
                rank_score=score,
                confidence=coverage,
                feature_schema_hash=schema_hash,
                metadata={
                    "feature_set_version": feature_set_version,
                    "feature_names": list(feature_names),
                    "vector_id": str(vector.vector_id),
                    "vector_as_of": vector.as_of.isoformat(),
                },
            )
        )
    return records, {
        "source": source,
        "feature_set_version": feature_set_version,
        "model_version": model_version,
        "feature_schema_hash": schema_hash,
        "required_features": list(feature_names),
        "vectors_loaded": len(vectors),
        "exact_vectors": len(exact_vectors),
        "records_prepared": len(records),
        "blockers": blockers,
    }


async def _xgboost_predictions(
    settings: PlatformSettings,
    *,
    feature_repo: FeatureRepository,
    instrument_ids: Sequence[uuid.UUID],
    contracts: Mapping[uuid.UUID, Mapping[str, object]],
    as_of: datetime,
    horizon: str,
    strategy_run: StrategyRun,
    manifest_path: Path | None,
) -> tuple[list[PredictionResult], dict[str, object]]:
    manifest = manifest_path or (
        Path(settings.boosting.artifact_manifest) if settings.boosting.artifact_manifest else None
    )
    if manifest is None:
        return [], {
            "source": "xgboost",
            "records_prepared": 0,
            "blockers": ["xgboost forecast materialization requires --xgboost-manifest"],
        }
    model = _load_xgboost_model(settings, manifest)
    vectors = await feature_repo.get_vectors(list(instrument_ids), model.feature_set_version, as_of)
    exact_vectors = {
        vector.instrument_id: vector
        for vector in vectors
        if _ensure_utc(vector.as_of) == as_of
        and _ensure_utc(vector.available_at or vector.as_of) <= as_of
    }
    blockers = _missing_vector_blockers(
        "xgboost",
        instrument_ids=instrument_ids,
        vectors=exact_vectors,
        contracts=contracts,
    )
    ordered_vectors = [
        exact_vectors[instrument_id]
        for instrument_id in instrument_ids
        if instrument_id in exact_vectors
    ]
    records: list[PredictionResult] = []
    if ordered_vectors:
        try:
            scores = model.score(ordered_vectors, strategy_run)
        except Exception as exc:
            blockers.append(f"xgboost scoring failed: {exc}")
            scores = []
        for score, vector in zip(scores, ordered_vectors, strict=True):
            coverage = float(model.feature_coverage(vector.features))
            records.append(
                _prediction_result(
                    source="xgboost",
                    model_version=model.model_version,
                    instrument_id=score.instrument_id,
                    strategy_run_id=strategy_run.run_id,
                    as_of=as_of,
                    horizon=horizon,
                    expected_return=float(score.score),
                    rank_score=float(score.score),
                    confidence=coverage,
                    feature_schema_hash=model.feature_schema_hash,
                    metadata={
                        "feature_set_version": model.feature_set_version,
                        "feature_names": list(model.feature_names),
                        "vector_id": str(vector.vector_id),
                        "vector_as_of": vector.as_of.isoformat(),
                        "device": model.device,
                        "manifest": str(manifest),
                    },
                )
            )
    return records, {
        "source": "xgboost",
        "feature_set_version": model.feature_set_version,
        "model_version": model.model_version,
        "feature_schema_hash": model.feature_schema_hash,
        "required_features": list(model.feature_names),
        "vectors_loaded": len(vectors),
        "exact_vectors": len(exact_vectors),
        "records_prepared": len(records),
        "blockers": blockers,
    }


def _load_xgboost_model(
    settings: PlatformSettings,
    manifest_path: Path,
) -> XGBoostRankSignalModel:
    from quant_platform.services.research_service.boosting.model import XGBoostRankSignalModel

    return XGBoostRankSignalModel(
        manifest_path,
        device=settings.boosting.device,
        require_gpu=settings.boosting.require_gpu,
    )


def _missing_vector_blockers(
    source: str,
    *,
    instrument_ids: Sequence[uuid.UUID],
    vectors: Mapping[uuid.UUID, FeatureVector],
    contracts: Mapping[uuid.UUID, Mapping[str, object]],
) -> list[str]:
    missing = [instrument_id for instrument_id in instrument_ids if instrument_id not in vectors]
    if not missing:
        return []
    symbols = ", ".join(symbol_for(contracts, instrument_id) for instrument_id in missing[:10])
    suffix = "" if len(missing) <= 10 else f", +{len(missing) - 10} more"
    return [
        f"{source} missing exact-as-of vectors for {len(missing)} instruments: {symbols}{suffix}"
    ]
