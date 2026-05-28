"""Governed ensemble signal model with source attribution."""

from __future__ import annotations

import math
import uuid
from typing import TYPE_CHECKING, Protocol

from quant_platform.core.domain.production import SignalContribution
from quant_platform.core.domain.signals import SignalScore
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.telemetry.metrics import (
    record_alpha_source_error,
    set_alpha_live_controls,
    set_alpha_source_coverage,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.core.domain.research import FeatureVector, StrategyRun


class SignalSourceModel(Protocol):
    def score(
        self,
        vectors: list[FeatureVector],
        strategy_run: StrategyRun,
    ) -> list[SignalScore]: ...


class MissingPromotedSignalSourceError(RuntimeError):
    """Raised when a promoted ensemble source cannot produce live scores."""


class EnsembleSignalModel:
    """Blend source signal models behind the existing SignalModel contract."""

    def __init__(
        self,
        *,
        sources: Mapping[str, SignalSourceModel],
        source_weights: Mapping[str, float],
        mode: str,
        max_non_classical_weight: float,
        fail_closed: bool = True,
        text_required_features: set[str] | None = None,
        required_features_by_source: Mapping[str, set[str]] | None = None,
        model_version: str = "ensemble-v1",
    ) -> None:
        self._sources = dict(sources)
        self._weights = _normalise_weights(source_weights, max_non_classical_weight, mode)
        self._mode = mode
        self._fail_closed = fail_closed
        self._required_features_by_source = {
            str(source): set(features)
            for source, features in (required_features_by_source or {}).items()
        }
        if text_required_features:
            self._required_features_by_source.setdefault("text", set()).update(
                text_required_features
            )
        self._model_version = model_version
        self.last_contributions: list[SignalContribution] = []
        if mode == "live":
            set_alpha_live_controls(
                cap=max_non_classical_weight,
                ramp_level=max_non_classical_weight,
            )

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def prediction_feature_names_by_source(self) -> dict[str, tuple[str, ...]]:
        feature_names_by_source: dict[str, tuple[str, ...]] = {}
        for source, model in self._sources.items():
            raw_feature_names = getattr(model, "feature_names", None)
            if callable(raw_feature_names):
                raw_feature_names = raw_feature_names()
            if raw_feature_names:
                feature_names_by_source[source] = tuple(str(name) for name in raw_feature_names)
                continue
            required = self._required_features_by_source.get(source)
            if required:
                feature_names_by_source[source] = tuple(sorted(required))
        return feature_names_by_source

    def score(
        self,
        vectors: list[FeatureVector],
        strategy_run: StrategyRun,
    ) -> list[SignalScore]:
        self.last_contributions = []
        if not vectors:
            return []

        source_scores: dict[str, dict[uuid.UUID, SignalScore]] = {}
        errors: list[str] = []
        for source, weight in self._weights.items():
            if weight <= 0:
                continue
            model = self._sources.get(source)
            if model is None:
                errors.append(f"missing model for promoted source {source!r}")
                record_alpha_source_error(source, "missing_model")
                continue
            # Feature schema version check is always a hard error — keep outside try block.
            if isinstance(model, LinearWeightSignalModel):
                expected_fsv = model.expected_feature_set_version
                if expected_fsv is not None:
                    for vec in vectors:
                        if vec.feature_set_version != expected_fsv:
                            raise ValueError(
                                f"ensemble source {source!r}: feature_set_version mismatch "
                                f"for instrument {vec.instrument_id}: "
                                f"expected {expected_fsv!r}, got {vec.feature_set_version!r}"
                            )
            try:
                self._assert_required_features_present(source, vectors)
                scores = model.score(vectors, strategy_run)
            except Exception as exc:
                errors.append(f"{source}: {exc}")
                record_alpha_source_error(source, type(exc).__name__)
                continue
            source_scores[source] = {score.instrument_id: score for score in scores}
            set_alpha_source_coverage(
                source,
                self._mode,
                len(source_scores[source]) / max(1, len(vectors)),
            )

        if errors and (self._mode in {"paper", "live"} and self._fail_closed):
            raise MissingPromotedSignalSourceError("; ".join(errors))

        ensemble_scores: list[SignalScore] = []
        for vec in vectors:
            raw = 0.0
            confidence_weighted_sum = 0.0
            covered_weight_sum = 0.0
            contribution_items: list[tuple[str, float, SignalScore]] = []
            for source, weight in self._weights.items():
                if weight <= 0:
                    continue
                score = source_scores.get(source, {}).get(vec.instrument_id)
                if score is None:
                    if self._mode in {"paper", "live"} and self._fail_closed:
                        raise MissingPromotedSignalSourceError(
                            f"source {source!r} did not score instrument {vec.instrument_id}"
                        )
                    continue
                raw += weight * score.score
                confidence_weighted_sum += score.confidence * weight
                covered_weight_sum += weight
                contribution_items.append((source, weight, score))

            clamped = max(-1.0, min(1.0, raw))
            # Weighted *average* confidence over the sources that actually
            # scored this vector. The prior code summed weighted parts and
            # clipped at 1.0, which silently masked overflow whenever weights
            # did not sum to 1 (common in shadow mode where the classical
            # source carries the bulk of weight while text/xgb are pinned to
            # small fractions). The weighted average is a true probability
            # and is mathematically bounded in [0, 1] without clipping.
            if covered_weight_sum > 0:
                confidence = confidence_weighted_sum / covered_weight_sum
            else:
                confidence = 0.0
            if not 0.0 <= confidence <= 1.0 + 1e-9:
                raise ValueError(
                    f"ensemble confidence {confidence} out of range — likely a"
                    " source produced confidence > 1.0"
                )
            confidence = max(0.0, min(1.0, confidence))
            score_id = uuid.uuid4()
            ensemble = SignalScore(
                score_id=score_id,
                instrument_id=vec.instrument_id,
                strategy_run_id=strategy_run.run_id,
                as_of=vec.as_of,
                score=clamped,
                confidence=confidence,
                model_version=self._model_version,
                feature_vector_id=vec.vector_id,
            )
            ensemble_scores.append(ensemble)
            for source, weight, source_score in contribution_items:
                self.last_contributions.append(
                    SignalContribution(
                        contribution_id=uuid.uuid4(),
                        score_id=score_id,
                        strategy_run_id=strategy_run.run_id,
                        instrument_id=vec.instrument_id,
                        as_of=vec.as_of,
                        source=source,
                        source_model_version=source_score.model_version,
                        raw_score=source_score.score,
                        normalized_score=source_score.score,
                        blend_weight=weight,
                        confidence=source_score.confidence,
                        feature_vector_id=source_score.feature_vector_id,
                        promotion_state=self._mode,
                    )
                )
        return ensemble_scores

    def _assert_required_features_present(
        self,
        source: str,
        vectors: list[FeatureVector],
    ) -> None:
        required = self._required_features_by_source.get(source, set())
        if not required:
            return
        for vec in vectors:
            missing = [name for name in sorted(required) if name not in vec.features]
            if missing:
                raise MissingPromotedSignalSourceError(
                    f"{source} source missing required features for {vec.instrument_id}: "
                    f"{', '.join(missing)}"
                )
            non_finite = [
                name for name in sorted(required) if not math.isfinite(float(vec.features[name]))
            ]
            if non_finite:
                raise MissingPromotedSignalSourceError(
                    f"{source} source has non-finite required features for "
                    f"{vec.instrument_id}: {', '.join(non_finite)}"
                )


def build_default_ensemble(
    *,
    classical_model: LinearWeightSignalModel,
    text_model: LinearWeightSignalModel | None,
    xgboost_model: SignalSourceModel | None,
    event_model: LinearWeightSignalModel | None,
    intraday_model: LinearWeightSignalModel | None,
    source_weights: Mapping[str, float],
    mode: str,
    max_non_classical_weight: float,
    fail_closed: bool,
    text_required_features: set[str],
    required_features_by_source: Mapping[str, set[str]] | None = None,
    model_version: str = "ensemble-v1",
) -> EnsembleSignalModel:
    sources: dict[str, SignalSourceModel] = {"classical": classical_model}
    if xgboost_model is not None:
        sources["xgboost"] = xgboost_model
    if text_model is not None:
        sources["text"] = text_model
    if event_model is not None:
        sources["event"] = event_model
    if intraday_model is not None:
        sources["intraday"] = intraday_model
    return EnsembleSignalModel(
        sources=sources,
        source_weights=source_weights,
        mode=mode,
        max_non_classical_weight=max_non_classical_weight,
        fail_closed=fail_closed,
        text_required_features=text_required_features,
        required_features_by_source=required_features_by_source,
        model_version=model_version,
    )


def _normalise_weights(
    source_weights: Mapping[str, float],
    max_non_classical_weight: float,
    mode: str,
) -> dict[str, float]:
    weights = {name: max(0.0, float(weight)) for name, weight in source_weights.items()}
    total = sum(weights.values())
    if total <= 0:
        return {"classical": 1.0}
    weights = {name: weight / total for name, weight in weights.items()}
    if mode in {"paper", "live"}:
        non_classical_sources = [name for name in weights if name not in {"classical", "primary"}]
        non_classical = sum(weights.get(name, 0.0) for name in non_classical_sources)
        cap = max(0.0, min(1.0, max_non_classical_weight))
        if non_classical > cap and non_classical > 0:
            scale = cap / non_classical
            for name in non_classical_sources:
                weights[name] = weights.get(name, 0.0) * scale
            if "classical" in weights:
                weights["classical"] = 1.0 - sum(
                    weights.get(name, 0.0) for name in non_classical_sources
                )
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items() if weight > 0}
