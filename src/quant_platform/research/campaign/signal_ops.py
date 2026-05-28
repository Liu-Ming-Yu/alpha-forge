"""Signal-gate recording helpers for research campaigns."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.bootstrap.governance.repositories import build_performance_repository
from quant_platform.core.domain.production import PredictionResult, SignalGateRecord
from quant_platform.services.governance_service.gates.signal_gate import (
    record_signal_observation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.production import SignalGateStatus
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

__all__ = ["record_campaign_prediction_evidence", "record_campaign_signal_gates"]


async def record_campaign_signal_gates(
    settings: PlatformSettings,
    *,
    model_version: str,
    train_xgboost: bool,
    signal_type: str | None = None,
    as_of: datetime,
    daily_ic: float,
    observations: int,
    drawdown: float,
    turnover: float,
    source_weights: Mapping[str, float] | None = None,
    daily_ics: Sequence[tuple[str, float]] | None = None,
) -> tuple[SignalGateStatus, SignalGateStatus]:
    """Record model-specific and source-level signal-gate evidence."""
    effective_signal_type = signal_type or ("xgboost" if train_xgboost else "classical")
    as_of_utc = _ensure_utc(as_of)
    gate = (
        build_performance_repository(settings.storage.postgres_dsn)
        if hasattr(settings, "storage")
        else None
    )
    model_status = await _record_signal_observations(
        settings=settings,
        gate=gate,
        signal_name=model_version,
        signal_type=effective_signal_type,
        as_of=as_of_utc,
        daily_ic=daily_ic,
        observations=observations,
        drawdown=drawdown,
        turnover=turnover,
        daily_ics=daily_ics,
    )
    if model_version == effective_signal_type:
        return model_status, model_status

    source_status = await _record_signal_observations(
        settings=settings,
        gate=gate,
        signal_name=effective_signal_type,
        signal_type=effective_signal_type,
        as_of=as_of_utc,
        daily_ic=daily_ic,
        observations=observations,
        drawdown=drawdown,
        turnover=turnover,
        daily_ics=daily_ics,
    )
    return source_status, model_status


async def record_campaign_prediction_evidence(
    settings: PlatformSettings,
    *,
    samples: Sequence[SupervisedAlphaSample],
    source_weights: Mapping[str, float],
    signal_type: str | None = None,
    model_version: str,
    feature_set_version: str,
    as_of: datetime,
    selected_weights: Mapping[str, float],
    horizon: str = "21d",
) -> dict[str, int]:
    """Persist source-specific forecast evidence from a governed campaign sample."""
    repo = build_performance_repository(settings.storage.postgres_dsn)
    strategy_run_id = uuid.uuid4()
    campaign_as_of_utc = _ensure_utc(as_of)
    counts: dict[str, int] = {}
    sources = tuple(
        str(source)
        for source, weight in source_weights.items()
        if float(weight) > 0.0 and str(source) not in {"classical", "primary"}
    )
    if signal_type in sources:
        sources = (str(signal_type),)
    for source in sources:
        feature_weights = _source_feature_weights(
            settings,
            source=source,
            selected_weights=selected_weights,
        )
        if not feature_weights:
            continue
        feature_names = tuple(feature_weights)
        schema_hash = ordered_feature_schema_hash(feature_names)
        saved = 0
        for sample in samples:
            confidence = _feature_coverage(sample.features, feature_names)
            if confidence <= 0.0:
                continue
            score = _linear_score(sample.features, feature_weights)
            sample_as_of = _ensure_utc(sample.as_of)
            await repo.save_prediction_result(
                PredictionResult(
                    prediction_id=uuid.uuid4(),
                    strategy_run_id=strategy_run_id,
                    instrument_id=sample.instrument_id,
                    source=f"campaign:{source}",
                    model_version=f"{model_version}:{source}",
                    as_of=sample_as_of,
                    horizon=horizon,
                    expected_return=score,
                    rank_score=score,
                    confidence=confidence,
                    feature_schema_hash=schema_hash,
                    calibration_bucket="research-campaign",
                    blockers=("offline_campaign_prediction_only",),
                    metadata={
                        "production_source": source,
                        "evidence_scope": "offline_campaign",
                        "feature_set_version": feature_set_version,
                        "campaign_as_of": campaign_as_of_utc.isoformat(),
                        "sample_as_of": sample_as_of.isoformat(),
                    },
                )
            )
            saved += 1
        counts[source] = saved
    return counts


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


async def _record_signal_observations(
    *,
    settings: PlatformSettings,
    gate: Any | None,
    signal_name: str,
    signal_type: str,
    as_of: datetime,
    daily_ic: float,
    observations: int,
    drawdown: float,
    turnover: float,
    daily_ics: Sequence[tuple[str, float]] | None,
) -> SignalGateStatus:
    if not daily_ics:
        kwargs: dict[str, Any] = {
            "signal_name": signal_name,
            "signal_type": signal_type,
            "as_of": as_of,
            "daily_ic": daily_ic,
            "observations": observations,
            "drawdown": drawdown,
            "turnover": turnover,
        }
        if gate is not None:
            kwargs["gate"] = gate
        return await record_signal_observation(
            settings,
            **kwargs,
        )
    if gate is None:
        gate = build_performance_repository(settings.storage.postgres_dsn)
    for raw_as_of, raw_ic in daily_ics:
        await gate.record_signal_observation(
            SignalGateRecord(
                signal_name=signal_name,
                signal_type=signal_type,
                as_of=_parse_daily_ic_as_of(raw_as_of),
                daily_ic=float(raw_ic),
                observations=1,
                drawdown=drawdown,
                turnover=turnover,
            )
        )
    return await gate.signal_status(
        signal_name,
        signal_type,
        as_of=as_of,
        min_observations=settings.production.text_gate_min_observations,
        min_ic=settings.production.text_gate_min_ic,
        max_negative_streak=settings.production.text_gate_max_negative_streak,
        drawdown_limit=settings.production.signal_gate_max_drawdown,
        turnover_limit=settings.production.signal_gate_max_turnover,
    )


def _parse_daily_ic_as_of(raw: str) -> datetime:
    value = datetime.fromisoformat(str(raw))
    return _ensure_utc(value)


def _source_feature_weights(
    settings: PlatformSettings,
    *,
    source: str,
    selected_weights: Mapping[str, float],
) -> dict[str, float]:
    if source == "text":
        return dict(settings.llm.text_feature_weights)
    if source == "event":
        return dict(settings.alpha.event_feature_weights)
    if source == "intraday":
        return dict(settings.alpha.intraday_feature_weights)
    if source == "xgboost":
        return {str(name): float(weight) for name, weight in selected_weights.items()}
    return {}


def _feature_coverage(features: Mapping[str, object], feature_names: Sequence[str]) -> float:
    if not feature_names:
        return 0.0
    return sum(1 for name in feature_names if name in features) / len(feature_names)


def _coerce_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return numeric


def _linear_score(features: Mapping[str, object], weights: Mapping[str, float]) -> float:
    denominator = sum(abs(float(weight)) for weight in weights.values()) or 1.0
    raw = 0.0
    for name, weight in weights.items():
        value = _coerce_float(features.get(name, 0.0))
        raw += value * float(weight)
    score = raw / denominator
    return max(-1.0, min(1.0, score))
