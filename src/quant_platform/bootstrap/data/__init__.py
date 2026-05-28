"""Data CLI operation composition helpers.

This module owns concrete data-service, broker, storage, and session wiring for
operator data commands.  CLI modules pass already-parsed request values here and
handle presentation/exit codes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from datetime import time as datetime_time
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from quant_platform.bootstrap.data.ingest import ingest_bars
from quant_platform.bootstrap.data.intraday import run_intraday_command
from quant_platform.bootstrap.persistence.migrations import (
    verify_database_head,
    verify_postgres_schema,
)
from quant_platform.bootstrap.session.public_api import create_paper_session
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)

__all__ = [
    "compute_features",
    "data_health_payload_for_contracts",
    "ingest_bars",
    "load_intraday_feature_series",
    "maintain_data",
    "reprocess_corporate_actions",
    "run_intraday_command",
]


async def load_intraday_feature_series(
    settings: PlatformSettings,
    contracts: Mapping[uuid.UUID, dict[str, object]],
    feature_set_version: str,
    decision_times: tuple[datetime, ...],
) -> tuple[
    dict[datetime, dict[uuid.UUID, dict[str, float]]],
    dict[datetime, datetime],
]:
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("0"),
        instrument_contracts=dict(contracts),
    )
    instrument_ids = list(contracts)
    feature_series: dict[datetime, dict[uuid.UUID, dict[str, float]]] = {}
    feature_available_at: dict[datetime, datetime] = {}
    for ts in decision_times:
        vectors = await session.feature_repo.get_vectors(
            instrument_ids,
            feature_set_version,
            ts,
        )
        if vectors:
            feature_series[ts] = {
                vector.instrument_id: {
                    name: float(value) for name, value in vector.features.items()
                }
                for vector in vectors
            }
            latest_available = max(
                (vector.available_at or vector.as_of for vector in vectors),
                default=ts,
            )
            feature_available_at[ts] = latest_available
        else:
            # Empty feature snapshots still carry explicit PIT availability.
            feature_available_at[ts] = ts
    return feature_series, feature_available_at


async def compute_features(
    settings: PlatformSettings,
    *,
    instrument_contracts: Mapping[uuid.UUID, dict[str, object]] | None = None,
) -> None:
    from quant_platform.bootstrap.data.feature_plugins import build_feature_registry
    from quant_platform.infrastructure.support.clock import WallClock
    from quant_platform.services.data_service.maintenance.maintenance_scheduler import (
        DataMaintenanceScheduler,
    )

    clock = WallClock()
    await verify_postgres_schema(settings)
    backend = "postgres" if settings.storage.postgres_dsn else "in_memory"
    contracts = dict(instrument_contracts or {})

    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("0"),
        instrument_contracts=contracts or None,
    )

    run = _make_strategy_run(settings)
    scheduler = DataMaintenanceScheduler(
        instruments=session.contract_master.list_active(),
        bar_store=session.bar_store,
        universe_manager=session.universe_manager,
        feature_repo=session.feature_repo,
        feature_registry=build_feature_registry(session.feature_repo),
    )
    result = await scheduler.run_once(
        strategy_run_id=run.run_id,
        as_of=clock.now(),
    )
    log.info(
        "compute_features.result",
        instruments=len(result.feature_data),
        bars_ingested=result.bars_ingested,
        liquidity_profiles=result.liquidity_profiles_updated,
        features_stored=result.features_stored,
        contracts=len(contracts),
        storage_backend=backend,
        object_store_root=str(Path(settings.storage.object_store_root).resolve()),
    )


async def maintain_data(
    settings: PlatformSettings,
    *,
    interval_seconds: float,
    backfill_start: date | None = None,
    backfill_end: date | None = None,
    instrument_contracts: Mapping[uuid.UUID, dict[str, object]] | None = None,
) -> None:
    """Run the DataMaintenanceSupervisor."""
    from quant_platform.bootstrap.data.feature_plugins import build_feature_registry
    from quant_platform.infrastructure.postgres.model_registry import build_model_registry
    from quant_platform.services.data_service.feeds.ingest_bar_fetcher_factory import (
        build_ingest_bar_fetcher,
    )
    from quant_platform.services.data_service.maintenance.maintenance_supervisor import (
        build_supervisor_for_paper,
    )

    backfill_mode = backfill_start is not None and backfill_end is not None

    if settings.storage.postgres_dsn:
        await verify_database_head(settings.storage.postgres_dsn)

    paper_session = create_paper_session(
        settings,
        initial_cash=Decimal("0"),
        instrument_contracts=dict(instrument_contracts or {}) or None,
    )
    await paper_session.broker.connect()
    try:
        strategy_run = _make_strategy_run(settings)
        registry = build_model_registry(settings.storage.postgres_dsn)
        bar_fetcher = build_ingest_bar_fetcher(settings, paper_session.broker)

        supervisor = await build_supervisor_for_paper(
            instruments=paper_session.contract_master.list_active(),
            bar_store=paper_session.bar_store,
            universe_manager=paper_session.universe_manager,
            feature_repo=paper_session.feature_repo,
            model_registry=registry,
            strategy_run_id=strategy_run.run_id,
            interval_seconds=interval_seconds,
            bar_fetcher=bar_fetcher,
            feature_registry=build_feature_registry(paper_session.feature_repo),
        )
        if backfill_mode:
            if backfill_start is None or backfill_end is None:
                raise RuntimeError("backfill start and end must be supplied for backfill mode")
            result = await supervisor.backfill_once(backfill_start, backfill_end)
            log.info(
                "maintain.backfill_complete",
                bars_fetched=result.bars_fetched,
                bars_stored=result.bars_stored,
                profiles=result.liquidity_profiles_updated,
                warnings=len(result.quality_warnings or []),
            )
            return
        await supervisor.run_forever(interval_seconds)
    finally:
        await paper_session.broker.disconnect()


async def reprocess_corporate_actions(
    settings: PlatformSettings,
    *,
    instrument_id: uuid.UUID,
) -> None:
    from quant_platform.services.data_service.stores.parquet_bar_store import ParquetBarStore

    store = ParquetBarStore(settings.storage.object_store_root)
    bars_written = await store.reprocess_corporate_actions(instrument_id)
    log.info(
        "reprocess_ca.complete",
        instrument_id=str(instrument_id),
        bars_written=bars_written,
    )


async def data_health_payload_for_contracts(
    settings: PlatformSettings,
    *,
    contracts: Mapping[uuid.UUID, dict[str, object]],
    start: date,
    end: date,
    bar_seconds: int,
) -> tuple[dict[str, object], bool]:
    from quant_platform.services.governance_service.gates.data_health import (
        build_data_health_report,
        data_health_payload,
    )

    await verify_postgres_schema(settings)
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("0"),
        instrument_contracts=dict(contracts),
    )
    start_dt = datetime.combine(start, datetime_time.min, tzinfo=UTC)
    end_dt = datetime.combine(end, datetime_time.max, tzinfo=UTC)
    report = await build_data_health_report(
        instruments=session.contract_master.list_active(),
        bar_store=session.bar_store,
        universe_manager=session.universe_manager,
        start=start_dt,
        end=end_dt,
        bar_seconds=bar_seconds,
        stale_after_days=settings.production.data_health_stale_after_days,
    )
    passed = not (
        report.coverage_pct < settings.production.data_health_min_coverage_pct
        or report.liquidity_coverage_pct
        < settings.production.data_health_min_liquidity_coverage_pct
        or report.stale_instruments > 0
    )
    return data_health_payload(report), passed


def _make_strategy_run(settings: PlatformSettings) -> StrategyRun:
    now = datetime.now(tz=UTC)
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="cli_cycle",
        strategy_version="0.1.0",
        run_type=RunType.PAPER if settings.broker.paper_trading else RunType.LIVE,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=now,
        started_at=now,
    )
