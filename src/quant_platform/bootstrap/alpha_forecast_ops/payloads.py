"""Shared forecast materialization payload and scoring helpers."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from quant_platform.core.domain.production import PredictionResult
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_FORECAST_NAMESPACE = uuid.UUID("b4b7cfd8-71d8-4f6d-b9f5-6f4df2f5638a")


def _prediction_result(
    *,
    source: str,
    model_version: str,
    instrument_id: uuid.UUID,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    horizon: str,
    expected_return: float,
    rank_score: float,
    confidence: float,
    feature_schema_hash: str,
    metadata: dict[str, object],
) -> PredictionResult:
    safe_confidence = max(0.0, min(1.0, float(confidence)))
    return PredictionResult(
        prediction_id=_stable_uuid(
            "prediction",
            source,
            model_version,
            str(instrument_id),
            as_of.isoformat(),
            horizon,
            feature_schema_hash,
        ),
        strategy_run_id=strategy_run_id,
        instrument_id=instrument_id,
        source=source,
        model_version=model_version,
        as_of=as_of,
        horizon=horizon,
        expected_return=float(expected_return),
        rank_score=float(rank_score),
        confidence=safe_confidence,
        feature_schema_hash=feature_schema_hash,
        calibration_bucket="current-promoted-feature-forecast",
        blockers=(),
        metadata=metadata,
    )


def _strategy_run(
    as_of: datetime,
    sources: Sequence[str],
    horizon: str,
) -> StrategyRun:
    run_id = _stable_uuid("strategy-run", as_of.isoformat(), ",".join(sources), horizon)
    return StrategyRun(
        run_id=run_id,
        strategy_name="alpha_materialize_forecasts",
        strategy_version="paper-v1",
        run_type=RunType.PAPER,
        status=RunStatus.COMPLETED,
        config_snapshot={"sources": list(sources), "horizon": horizon},
        created_at=as_of,
        started_at=as_of,
        finished_at=as_of,
    )


def _stable_uuid(*parts: object) -> uuid.UUID:
    return uuid.uuid5(_FORECAST_NAMESPACE, "|".join(str(part) for part in parts))


def _feature_coverage(features: Mapping[str, object], feature_names: Sequence[str]) -> float:
    if not feature_names:
        return 0.0
    return sum(1 for name in feature_names if name in features) / len(feature_names)


def _payload_blockers(payload: Mapping[str, object]) -> list[str]:
    raw = payload.get("blockers", ())
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw]
    return []


def _coerce_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def _linear_score(features: Mapping[str, object], weights: Mapping[str, float]) -> float:
    denominator = sum(abs(float(weight)) for weight in weights.values()) or 1.0
    raw = 0.0
    for name, weight in weights.items():
        value = _coerce_float(features.get(name, 0.0))
        raw += value * float(weight)
    score = raw / denominator
    return max(-1.0, min(1.0, score))


def symbol_for(
    contracts: Mapping[uuid.UUID, Mapping[str, object]],
    instrument_id: uuid.UUID,
) -> str:
    symbol = str(contracts.get(instrument_id, {}).get("symbol", "")).strip()
    return symbol or str(instrument_id)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _payload(
    *,
    passed: bool,
    as_of: datetime,
    horizon: str,
    sources: Sequence[str],
    source_payloads: Sequence[Mapping[str, object]],
    blockers: Sequence[str],
    saved: int,
    reason: str,
) -> dict[str, object]:
    return {
        "passed": passed,
        "reason": reason,
        "as_of": as_of.isoformat(),
        "horizon": horizon,
        "sources": list(sources),
        "prediction_results_saved": saved,
        "blockers": list(blockers),
        "source_results": list(source_payloads),
    }
