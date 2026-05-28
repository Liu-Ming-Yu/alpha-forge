"""Signal orchestration controller.

GenerateSignalsControllerImpl wraps a SignalModel with in-memory feature
sourcing.  Callers supply raw feature dicts; this controller builds
FeatureVectors, runs the model, and emits SignalScorePublished events.

Research-to-production parity:
    The same controller is used in both paper and backtest contexts.  Only
    the feature data source differs, not the scoring logic.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING

import structlog

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.core.domain.production import PredictionResult, SignalContribution
from quant_platform.core.domain.research import FeatureVector, StrategyRun
from quant_platform.core.events import SignalScorePublished

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.contracts import (
        EventBus,
        PredictionEvidenceRepository,
        SignalContributionRepository,
        SignalModel,
    )
    from quant_platform.core.domain.signals import SignalScore

log = structlog.get_logger(__name__)


class GenerateSignalsControllerImpl:
    """Generate signal scores from caller-supplied feature data.

    This is the in-process, in-memory implementation of signal generation.
    Feature data is provided as plain dicts (no DB required), making it
    suitable for both paper-trading and backtest scenarios.

    Args:
        signal_model: Any SignalModel implementation
            (e.g. LinearWeightSignalModel).
        event_bus: EventBus for publishing SignalScorePublished events.
        feature_set_version: Version tag embedded in every FeatureVector
            created by this controller.

    Must never:
        Access the broker gateway or account state.
        Perform capital allocation or order construction.
    """

    def __init__(
        self,
        signal_model: SignalModel,
        event_bus: EventBus,
        feature_set_version: str = "1.0.0",
        signal_contribution_repo: SignalContributionRepository | None = None,
        prediction_evidence_repo: PredictionEvidenceRepository | None = None,
        prediction_horizon: str = "rank_1d",
        calibration_bucket: str = "rank_score_uncalibrated",
    ) -> None:
        self._model = signal_model
        self._bus = event_bus
        self._feature_set_version = feature_set_version
        self._signal_contribution_repo = signal_contribution_repo
        self._prediction_evidence_repo = prediction_evidence_repo
        self._prediction_horizon = prediction_horizon
        self._calibration_bucket = calibration_bucket

    async def generate(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],
        strategy_run: StrategyRun,
        as_of: datetime,
    ) -> list[SignalScore]:
        """Score feature data and emit one SignalScorePublished event per score.

        Args:
            feature_data: Mapping of instrument_id → {feature_name: value}.
                Instruments with no entry are silently skipped.
            strategy_run: The active StrategyRun, used for attribution on
                every FeatureVector and SignalScore created.
            as_of: UTC timestamp embedded in every FeatureVector.
                Must be timezone-aware.

        Returns:
            List of SignalScore objects, one per instrument in feature_data.
            Empty list if feature_data is empty.

        Raises:
            ValueError: as_of is not timezone-aware.
        """
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")

        if not feature_data:
            return []

        vectors: list[FeatureVector] = [
            FeatureVector(
                vector_id=uuid.uuid4(),
                instrument_id=instr_id,
                as_of=as_of,
                feature_set_version=self._feature_set_version,
                features=features,
                strategy_run_id=strategy_run.run_id,
            )
            for instr_id, features in feature_data.items()
        ]

        scores = self._model.score(vectors, strategy_run)

        contributions = getattr(self._model, "last_contributions", None)
        if contributions and self._signal_contribution_repo is not None:
            await self._signal_contribution_repo.save_signal_contributions(contributions)
        if contributions and self._prediction_evidence_repo is not None:
            vector_features = {vector.vector_id: dict(vector.features) for vector in vectors}
            feature_names_by_source = _model_prediction_feature_names_by_source(self._model)
            predictions = _prediction_results_from_contributions(
                contributions,
                vector_features,
                feature_names_by_source,
                horizon=self._prediction_horizon,
                calibration_bucket=self._calibration_bucket,
                feature_set_version=self._feature_set_version,
                ensemble_model_version=str(getattr(self._model, "model_version", "unknown")),
            )
            for prediction in predictions:
                await self._prediction_evidence_repo.save_prediction_result(prediction)

        for score in scores:
            await self._bus.publish(
                SignalScorePublished(
                    event_id=uuid.uuid4(),
                    occurred_at=as_of,
                    score_id=score.score_id,
                    instrument_id=score.instrument_id,
                    strategy_run_id=score.strategy_run_id,
                )
            )

        log.info(
            "signals.generated",
            count=len(scores),
            strategy_run_id=str(strategy_run.run_id),
        )
        return scores


def _prediction_results_from_contributions(
    contributions: list[SignalContribution],
    vector_features: dict[uuid.UUID, dict[str, float]],
    feature_names_by_source: Mapping[str, tuple[str, ...]],
    *,
    horizon: str,
    calibration_bucket: str,
    feature_set_version: str,
    ensemble_model_version: str,
) -> list[PredictionResult]:
    predictions: list[PredictionResult] = []
    for contribution in contributions:
        if contribution.source in {"classical", "primary"}:
            continue
        features = (
            vector_features.get(contribution.feature_vector_id)
            if contribution.feature_vector_id is not None
            else None
        )
        blockers: tuple[str, ...] = ()
        feature_names = feature_names_by_source.get(contribution.source)
        schema_source = "source_model"
        if not feature_names:
            feature_names = (
                tuple(sorted(features)) if features else (f"feature_set:{feature_set_version}",)
            )
            schema_source = "full_vector_fallback"
            blockers = ("source_feature_schema_unknown",)
        predictions.append(
            PredictionResult(
                prediction_id=uuid.uuid4(),
                strategy_run_id=contribution.strategy_run_id,
                instrument_id=contribution.instrument_id,
                source=contribution.source,
                model_version=contribution.source_model_version or f"{contribution.source}-unknown",
                as_of=contribution.as_of,
                horizon=horizon,
                expected_return=contribution.normalized_score,
                rank_score=contribution.normalized_score,
                confidence=contribution.confidence,
                feature_schema_hash=ordered_feature_schema_hash(feature_names),
                calibration_bucket=calibration_bucket,
                blockers=blockers,
                metadata={
                    "expected_return_semantics": "rank_score_proxy",
                    "feature_set_version": feature_set_version,
                    "feature_schema_source": schema_source,
                    "feature_names": list(feature_names),
                    "ensemble_model_version": ensemble_model_version,
                    "ensemble_score_id": str(contribution.score_id),
                    "source_feature_vector_id": (
                        str(contribution.feature_vector_id)
                        if contribution.feature_vector_id is not None
                        else None
                    ),
                    "promotion_state": contribution.promotion_state,
                    "blend_weight": contribution.blend_weight,
                },
            )
        )
    return predictions


def _model_prediction_feature_names_by_source(
    model: object,
) -> dict[str, tuple[str, ...]]:
    raw = getattr(model, "prediction_feature_names_by_source", {})
    if callable(raw):
        raw = raw()
    if not isinstance(raw, Mapping):
        return {}
    feature_names_by_source: dict[str, tuple[str, ...]] = {}
    for source, feature_names in raw.items():
        if not isinstance(feature_names, tuple | list | set):
            continue
        feature_names_by_source[str(source)] = tuple(str(name) for name in feature_names)
    return feature_names_by_source
