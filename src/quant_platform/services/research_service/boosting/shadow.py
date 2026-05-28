"""Shadow scoring artifact writer for XGBoost signals."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import PredictionResult, SignalContribution
from quant_platform.core.domain.research import FeatureVector, StrategyRun

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.core.contracts import (
        PredictionEvidenceRepository,
        SignalContributionRepository,
    )
    from quant_platform.core.domain.signals import SignalScore
    from quant_platform.services.research_service.boosting.model import XGBoostRankSignalModel


class ShadowBoostingScorer:
    """Write boosted-tree shadow scores as JSONL without publishing signals."""

    def __init__(
        self,
        *,
        model: XGBoostRankSignalModel,
        artifact_root: Path | str,
        contribution_repo: SignalContributionRepository | None = None,
        prediction_evidence_repo: PredictionEvidenceRepository | None = None,
        horizon: str = "21d",
    ) -> None:
        self._model = model
        self._artifact_root = Path(artifact_root)
        self._contribution_repo = contribution_repo
        self._prediction_evidence_repo = prediction_evidence_repo
        self._horizon = horizon

    async def score_cycle(
        self,
        *,
        feature_data: Mapping[uuid.UUID, Mapping[str, float]],
        primary_scores: Sequence[SignalScore],
        strategy_run: StrategyRun,
        as_of: datetime,
    ) -> Path | None:
        if not feature_data:
            return None
        vectors = [
            FeatureVector(
                vector_id=uuid.uuid4(),
                instrument_id=instrument_id,
                as_of=as_of,
                feature_set_version=self._model.feature_set_version,
                features=dict(features),
                strategy_run_id=strategy_run.run_id,
            )
            for instrument_id, features in feature_data.items()
        ]
        boosted = self._model.score(vectors, strategy_run)
        primary_by_instrument = {score.instrument_id: score.score for score in primary_scores}
        contributions: list[SignalContribution] = []
        predictions: list[PredictionResult] = []

        self._artifact_root.mkdir(parents=True, exist_ok=True)
        path = self._artifact_root / f"{as_of.strftime('%Y%m%dT%H%M%S%z')}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for score, vector in zip(boosted, vectors, strict=True):
                feature_coverage = self._model.feature_coverage(vector.features)
                row = {
                    "as_of": as_of.astimezone(UTC).isoformat(),
                    "strategy_run_id": str(strategy_run.run_id),
                    "instrument_id": str(score.instrument_id),
                    "boosted_score": score.score,
                    "primary_score": primary_by_instrument.get(score.instrument_id),
                    "model_version": self._model.model_version,
                    "feature_set_version": self._model.feature_set_version,
                    "feature_schema_hash": self._model.feature_schema_hash,
                    "feature_coverage": feature_coverage,
                    "device": self._model.device,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                contributions.append(
                    SignalContribution(
                        contribution_id=uuid.uuid4(),
                        score_id=score.score_id,
                        strategy_run_id=strategy_run.run_id,
                        instrument_id=score.instrument_id,
                        as_of=score.as_of,
                        source="xgboost",
                        source_model_version=score.model_version,
                        raw_score=score.score,
                        normalized_score=score.score,
                        blend_weight=0.0,
                        confidence=score.confidence,
                        feature_vector_id=score.feature_vector_id,
                        promotion_state="shadow",
                    )
                )
                predictions.append(
                    PredictionResult(
                        prediction_id=uuid.uuid4(),
                        strategy_run_id=strategy_run.run_id,
                        instrument_id=score.instrument_id,
                        source="xgboost",
                        model_version=self._model.model_version,
                        as_of=score.as_of,
                        horizon=self._horizon,
                        expected_return=score.score,
                        rank_score=score.score,
                        confidence=score.confidence,
                        feature_schema_hash=self._model.feature_schema_hash,
                        calibration_bucket="shadow:daily",
                        metadata={
                            "device": self._model.device,
                            "feature_coverage": feature_coverage,
                            "feature_set_version": self._model.feature_set_version,
                        },
                    )
                )
        if contributions and self._contribution_repo is not None:
            await self._contribution_repo.save_signal_contributions(contributions)
        if predictions and self._prediction_evidence_repo is not None:
            for prediction in predictions:
                await self._prediction_evidence_repo.save_prediction_result(prediction)
        return path
