"""Durable historical feature-vector backfill operation."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.features.backfill import (
    FeatureBackfillDay,
    FeatureBackfillResult,
    run_feature_backfill,
)
from quant_platform.bootstrap.session.public_api import create_paper_session
from quant_platform.research.common import (
    _load_instrument_contracts,
    _require_durable_research_inputs,
    _verify_postgres_schema_if_configured,
    research_json_result,
)
from quant_platform.research.features.backfill_compute import (
    backfill_ohlcv_feature_set,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.application.research import FeaturesBackfillRequest
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)

__all__ = ["FeatureBackfillDay", "FeatureBackfillResult", "_features_backfill"]

_MIN_HISTORY_BARS = 21


async def _features_backfill(
    settings: PlatformSettings, request: FeaturesBackfillRequest
) -> UseCaseResult[dict[str, object]]:
    """Backfill point-in-time feature vectors into the durable feature repository."""
    result = await _run_feature_backfill(
        settings=settings,
        contracts_file=str(request.contracts_file),
        start=request.start,
        end=request.end,
        feature_set_version=str(request.feature_set_version),
        bar_seconds=int(request.bar_seconds),
        lookback_days=int(request.lookback_days),
        date_policy=request.date_policy,
        source_data_manifest=request.source_data_manifest,
        dry_run=bool(request.dry_run),
    )
    return research_json_result(result.to_payload())


async def _run_feature_backfill(
    *,
    settings: PlatformSettings,
    contracts_file: str,
    start: datetime,
    end: datetime,
    feature_set_version: str,
    bar_seconds: int,
    lookback_days: int,
    dry_run: bool,
    date_policy: str = "nyse-sessions",
    source_data_manifest: Path | None = None,
) -> FeatureBackfillResult:
    if lookback_days < _MIN_HISTORY_BARS:
        raise OperatorUsageError(f"--lookback-days must be >= {_MIN_HISTORY_BARS}")
    if bar_seconds <= 0:
        raise OperatorUsageError("--bar-seconds must be > 0")
    if end < start:
        raise OperatorUsageError("--end must be >= --start")

    _require_durable_research_inputs(settings)
    await _verify_postgres_schema_if_configured(settings)

    contracts = _load_instrument_contracts(contracts_file)
    events_by_instrument = _load_events_by_instrument(source_data_manifest)
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("0"),
        instrument_contracts=contracts,
    )
    result = await run_feature_backfill(
        instruments=session.contract_master.list_active(),
        bar_store=session.bar_store,
        feature_repo=session.feature_repo,
        artifact_uri=_feature_artifact_uri(settings),
        start=start,
        end=end,
        feature_set_version=feature_set_version,
        bar_seconds=bar_seconds,
        lookback_days=lookback_days,
        date_policy=date_policy,
        events_by_instrument=events_by_instrument,
        dry_run=dry_run,
        feature_set_backfiller=backfill_ohlcv_feature_set,
    )
    log.info("features_backfill.complete", **result.to_payload())
    return result


def _load_events_by_instrument(
    source_data_manifest: Path | None,
) -> dict[uuid.UUID, tuple[datetime, ...]] | None:
    if source_data_manifest is None:
        return None
    from quant_platform.services.research_service.events.candidates.screening import (
        events_by_instrument_from_manifest,
    )

    payload = json.loads(source_data_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OperatorUsageError("--source-data-manifest must contain a JSON object")
    return events_by_instrument_from_manifest(payload)


def _feature_artifact_uri(settings: PlatformSettings) -> str:
    return Path(settings.storage.object_store_root).resolve().as_uri()
