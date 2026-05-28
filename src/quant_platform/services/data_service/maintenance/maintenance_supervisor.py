"""DataMaintenanceSupervisor — independent loop driving feature jobs.

Decouples ingest/feature maintenance from the strategy cycle so bar ingest
and feature computation can run on their own cadence (typically every few
minutes) without blocking or being blocked by the rebalance loop.

Selection of model registry (in-memory vs Postgres) is handled by
:func:`quant_platform.infrastructure.postgres.model_registry.build_model_registry`.

CLI::

    python -m quant_platform maintain --interval 900
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Protocol, cast

import structlog

from quant_platform.services.data_service.maintenance.maintenance_backfill import (
    run_maintenance_backfill,
)
from quant_platform.services.data_service.maintenance.maintenance_jobs import (
    MaintenanceTick,
    maybe_await,
    run_due_feature_jobs,
)
from quant_platform.services.data_service.maintenance.maintenance_retention import (
    maybe_sweep_event_bus,
)
from quant_platform.services.data_service.maintenance.maintenance_scheduler import (
    DEFAULT_FEATURE_SET_VERSION,
    DataMaintenanceScheduler,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable

    from quant_platform.application.features.registry import FeatureFamilyRegistry
    from quant_platform.core.contracts import (
        Clock,
        FeatureRepository,
        HistoricalDataStore,
        MarketDataProvider,
    )
    from quant_platform.core.domain.instruments import Instrument
    from quant_platform.services.data_service.ingest.daily_ingest import (
        BarFetcher,
        IngestResult,
    )
    from quant_platform.services.data_service.maintenance.maintenance_retention import (
        EventBusRetentionWorker,
    )
    from quant_platform.services.data_service.reference.universe_manager import UniverseManager

log = structlog.get_logger(__name__)


class MaintenanceModelRegistry(Protocol):
    def register_model(
        self,
        *,
        strategy_name: str,
        model_version: str,
        feature_set_version: str,
        as_of: datetime,
        metadata: dict[str, object] | None = None,
    ) -> RegisteredModelRef | Awaitable[RegisteredModelRef]: ...

    def schedule_feature_job(
        self,
        *,
        model_id: uuid.UUID,
        strategy_name: str,
        feature_set_version: str,
        interval_seconds: float,
        as_of: datetime,
    ) -> object | Awaitable[object]: ...


class RegisteredModelRef(Protocol):
    """Minimal registered-model shape consumed by data maintenance."""

    @property
    def model_id(self) -> uuid.UUID: ...


class _UtcClock:
    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    def today(self) -> date:
        return self.now().date()


class DataMaintenanceSupervisor:
    """Periodic runner for :class:`DataMaintenanceScheduler`.

    Pulls due jobs from the configured model registry, executes them via the
    scheduler, and records completion (with exponential back-off via the
    registry's own ``mark_job_completed`` logic).

    The supervisor does not coordinate with the strategy cycle via a shared
    lock — feature writes are append-only and keyed by ``(instrument_id,
    feature_set_version, as_of)`` so concurrent runs do not corrupt state.
    """

    def __init__(
        self,
        *,
        model_registry: MaintenanceModelRegistry,
        scheduler: DataMaintenanceScheduler,
        strategy_run_id: uuid.UUID,
        clock: Clock | None = None,
        instruments: list[Instrument] | None = None,
        bar_store: HistoricalDataStore | None = None,
        universe_manager: UniverseManager | None = None,
        bar_fetcher: BarFetcher | None = None,
        event_bus_retention: EventBusRetentionWorker | None = None,
        event_bus_retention_interval_seconds: float = 0.0,
    ) -> None:
        self._registry = model_registry
        self._scheduler = scheduler
        self._strategy_run_id = strategy_run_id
        self._clock = clock or _UtcClock()
        self._instruments = instruments
        self._bar_store = bar_store
        self._universe_manager = universe_manager
        self._bar_fetcher = bar_fetcher
        # Optional Redis-Streams retention sweeper (Phase 4.2).  The
        # supervisor drives it at its own cadence so feature-job load
        # cannot starve retention.  ``None`` disables the sweep.
        self._event_bus_retention = event_bus_retention
        self._retention_interval = event_bus_retention_interval_seconds
        self._last_retention_sweep: float = 0.0

    async def _maybe_sweep_event_bus(self) -> None:
        """Run the XTRIM sweeper when it is due.

        Cadence is tracked in monotonic seconds so a wall-clock jump
        cannot either starve or stampede the sweep.
        """
        self._last_retention_sweep = await maybe_sweep_event_bus(
            worker=self._event_bus_retention,
            interval_seconds=self._retention_interval,
            last_sweep_monotonic=self._last_retention_sweep,
        )

    async def tick(self) -> MaintenanceTick:
        """Run one pass: fetch due jobs, execute each, record outcome."""
        as_of = self._clock.now()
        await self._maybe_sweep_event_bus()
        return await run_due_feature_jobs(
            as_of=as_of,
            model_registry=self._registry,
            scheduler=self._scheduler,
            strategy_run_id=self._strategy_run_id,
        )

    async def backfill_once(
        self,
        start: date,
        end: date,
    ) -> IngestResult:
        """One-shot bar-data backfill over ``[start, end]``.

        Iterates the registered universe and calls ``run_daily_ingest`` once
        using the configured ``bar_fetcher`` — independent of the
        feature-jobs cadence.  Useful as a cold-start or gap-recovery tool
        invoked via ``quant-platform maintain --backfill-start ...
        --backfill-end ...``.

        Raises:
            ValueError: When the supervisor was constructed without the
                ingest dependencies (``instruments``, ``bar_store``,
                ``universe_manager``, ``bar_fetcher``).  These are optional
                at construction so ``tick``-only deployments (e.g. the API
                server) do not have to supply a broker.
        """
        return await run_maintenance_backfill(
            instruments=self._instruments,
            bar_store=self._bar_store,
            universe_manager=self._universe_manager,
            bar_fetcher=self._bar_fetcher,
            start=start,
            end=end,
        )

    async def run_forever(self, interval_seconds: float) -> None:
        """Sleep-driven loop; exits cleanly on ``asyncio.CancelledError``.

        Shortens the sleep when a tick takes longer than ``interval_seconds``
        so maintenance cannot drift arbitrarily far behind wall-clock.
        """
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        log.info(
            "maintenance_supervisor.start",
            interval_seconds=interval_seconds,
            strategy_run_id=str(self._strategy_run_id),
        )
        try:
            while True:
                started = self._clock.now()
                summary = await self.tick()
                elapsed = (self._clock.now() - started).total_seconds()
                log.info(
                    "maintenance_supervisor.tick",
                    jobs_processed=summary.jobs_processed,
                    features_stored=summary.features_stored,
                    errors=summary.errors,
                    elapsed_seconds=elapsed,
                )
                sleep_for = max(0.0, interval_seconds - elapsed)
                await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            log.info("maintenance_supervisor.cancelled")
            raise


async def build_supervisor_for_paper(
    *,
    instruments: list[Instrument],
    bar_store: HistoricalDataStore,
    universe_manager: UniverseManager,
    feature_repo: FeatureRepository,
    model_registry: MaintenanceModelRegistry,
    strategy_run_id: uuid.UUID,
    market_data_provider: MarketDataProvider | None = None,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    interval_seconds: float = 900.0,
    as_of: datetime | None = None,
    bar_fetcher: BarFetcher | None = None,
    event_bus_retention: EventBusRetentionWorker | None = None,
    event_bus_retention_interval_seconds: float = 0.0,
    feature_registry: FeatureFamilyRegistry | None = None,
) -> DataMaintenanceSupervisor:
    """Factory: register a model + feature job and return a supervisor.

    Convenience for ``python -m quant_platform maintain`` and tests; production
    deployments typically register models explicitly during engine bootstrap
    (see :class:`EngineRunner`).
    """
    clock = _UtcClock()
    now = as_of or clock.now()
    model = cast(
        "RegisteredModelRef",
        await maybe_await(
            model_registry.register_model(
                strategy_name="maintenance_daemon",
                model_version="0.0.0",
                feature_set_version=feature_set_version,
                as_of=now,
            )
        ),
    )
    await maybe_await(
        model_registry.schedule_feature_job(
            model_id=model.model_id,
            strategy_name="maintenance_daemon",
            feature_set_version=feature_set_version,
            interval_seconds=interval_seconds,
            as_of=now,
        )
    )

    scheduler = DataMaintenanceScheduler(
        instruments=instruments,
        bar_store=bar_store,
        universe_manager=universe_manager,
        feature_repo=feature_repo,
        market_data_provider=market_data_provider,
        feature_registry=feature_registry,
    )
    return DataMaintenanceSupervisor(
        model_registry=model_registry,
        scheduler=scheduler,
        strategy_run_id=strategy_run_id,
        clock=clock,
        instruments=instruments,
        bar_store=bar_store,
        universe_manager=universe_manager,
        bar_fetcher=bar_fetcher,
        event_bus_retention=event_bus_retention,
        event_bus_retention_interval_seconds=event_bus_retention_interval_seconds,
    )
