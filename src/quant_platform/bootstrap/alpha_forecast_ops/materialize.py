"""Forecast evidence orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.research.common import _load_instrument_contracts
from quant_platform.bootstrap.alpha_forecast_ops.payloads import (
    _ensure_utc,
    _payload,
    _payload_blockers,
    _strategy_run,
)
from quant_platform.bootstrap.alpha_forecast_ops.policies import _normalise_sources
from quant_platform.bootstrap.alpha_forecast_ops.repositories import (
    build_alpha_forecast_feature_repository,
)
from quant_platform.bootstrap.alpha_forecast_ops.sources import (
    _linear_source_predictions,
    _xgboost_predictions,
)
from quant_platform.bootstrap.governance.repositories import build_performance_repository
from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.core.domain.production import PredictionResult


async def materialize_alpha_forecasts(
    settings: PlatformSettings,
    *,
    contracts_file: Path,
    as_of: datetime,
    sources: Sequence[str],
    horizon: str,
    xgboost_manifest: Path | None,
    fail_on_missing: bool,
    feature_repo: FeatureRepository | None = None,
) -> dict[str, object]:
    """Write deterministic ``PredictionResult`` rows for promoted paper sources."""
    if settings.storage.postgres_dsn:
        await verify_postgres_schema(settings)
    as_of_utc = _ensure_utc(as_of)
    selected_sources = _normalise_sources(sources)
    contracts = _load_instrument_contracts(str(contracts_file))
    instrument_ids = tuple(contracts)
    active_feature_repo = feature_repo or build_alpha_forecast_feature_repository(settings)

    records: list[PredictionResult] = []
    source_payloads: list[dict[str, object]] = []
    blockers: list[str] = []
    strategy_run = _strategy_run(as_of_utc, selected_sources, horizon)

    for source in selected_sources:
        if source == "xgboost":
            source_records, source_payload = await _xgboost_predictions(
                settings,
                feature_repo=active_feature_repo,
                instrument_ids=instrument_ids,
                contracts=contracts,
                as_of=as_of_utc,
                horizon=horizon,
                strategy_run=strategy_run,
                manifest_path=xgboost_manifest,
            )
        else:
            source_records, source_payload = await _linear_source_predictions(
                settings,
                source=source,
                feature_repo=active_feature_repo,
                instrument_ids=instrument_ids,
                contracts=contracts,
                as_of=as_of_utc,
                horizon=horizon,
                strategy_run=strategy_run,
            )
        source_payloads.append(source_payload)
        blockers.extend(_payload_blockers(source_payload))
        records.extend(source_records)

    if blockers:
        return _payload(
            passed=False,
            as_of=as_of_utc,
            horizon=horizon,
            sources=selected_sources,
            source_payloads=source_payloads,
            blockers=blockers,
            saved=0,
            reason="forecast materialization blocked before writing prediction evidence",
        )

    repo = build_performance_repository(settings.storage.postgres_dsn)
    for record in records:
        await repo.save_prediction_result(record)

    return _payload(
        passed=True,
        as_of=as_of_utc,
        horizon=horizon,
        sources=selected_sources,
        source_payloads=source_payloads,
        blockers=blockers,
        saved=len(records),
        reason="forecast materialization complete",
    )


__all__ = ["materialize_alpha_forecasts"]
