"""Cycle-dispatch behavior for ``EngineRunner``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.application.runtime.state import CycleResult
from quant_platform.engines.feature_jobs.job_runtime import run_due_feature_jobs
from quant_platform.engines.feature_jobs.schema_guard import validate_required_feature_schema
from quant_platform.engines.market_data.price_seeding import (
    build_cycle_market_prices,
    validate_positive_feature_prices,
)
from quant_platform.engines.proposals.v2_cycle import run_v2_proposal_cycle

if TYPE_CHECKING:
    import asyncio
    import uuid
    from collections.abc import Sequence
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.application.runtime.state import Session
    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.production import EngineTargetProposal
    from quant_platform.core.domain.research.runs import StrategyRun
    from quant_platform.core.domain.signals import SignalScore
    from quant_platform.engines.framework.types import EngineConfig, EngineRunResult
    from quant_platform.infrastructure.postgres.model_registry import PostgresModelRegistry
    from quant_platform.services.data_service.maintenance.maintenance_scheduler import (
        DataMaintenanceScheduler,
    )
    from quant_platform.services.research_service.modeling.registry.model_registry import (
        InMemoryModelRegistry,
    )

log = structlog.get_logger(__name__)


class EngineRunnerCycleMixin:
    """Feature loading and mode dispatch for one engine cycle."""

    _config: EngineConfig
    _settings: PlatformSettings
    _session: Session | None
    _strategy_run: StrategyRun | None
    _maintenance_scheduler: DataMaintenanceScheduler | None
    _model_registry: InMemoryModelRegistry | PostgresModelRegistry
    _cycle_lock: asyncio.Lock
    _result: EngineRunResult

    if TYPE_CHECKING:

        async def _run_shadow_cycle(
            self,
            feature_data: dict[uuid.UUID, dict[str, float]],
            market_prices: dict[uuid.UUID, Decimal] | None = None,
        ) -> CycleResult: ...

        async def _run_shadow_boosting_cycle(
            self,
            feature_data: dict[uuid.UUID, dict[str, float]],
            primary_scores: Sequence[SignalScore],
            as_of: datetime,
        ) -> None: ...

        async def _run_shadow_text_cycle(
            self,
            as_of: datetime,
            market_prices: dict[uuid.UUID, Decimal] | None = None,
        ) -> None: ...

        async def _generate_proposal_inner(
            self,
            feature_data: dict[uuid.UUID, dict[str, float]],
            market_prices: dict[uuid.UUID, Decimal] | None,
            as_of: datetime,
            *,
            feature_dataset_id: uuid.UUID | None = None,
        ) -> EngineTargetProposal: ...

    async def _scheduled_feature_data(self, as_of: datetime) -> dict[uuid.UUID, dict[str, float]]:
        return await run_due_feature_jobs(
            model_registry=self._model_registry,
            maintenance_scheduler=self._maintenance_scheduler,
            strategy_run=self._strategy_run,
            as_of=as_of,
            engine_name=self._config.engine_name,
            halt_on_stale_features=self._settings.risk.halt_on_stale_features,
            fail_on_error=self._config.uses_order_capable_external_broker,
        )

    async def run_cycle(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],
        market_prices: dict[uuid.UUID, Decimal] | None = None,
    ) -> CycleResult:
        """Execute one rebalance cycle."""
        from quant_platform.engines.framework.types import RunMode

        session = self._session
        strategy_run = self._strategy_run
        if session is None or strategy_run is None:
            raise RuntimeError("call initialize() first")

        if self._cycle_lock.locked():
            log.warning(
                "engine_runner.cycle_reentrancy_guard",
                engine=self._config.engine_name,
                detail="previous cycle still running; skipping this tick",
            )
            return CycleResult(
                signals=[], target=None, approved=[], rejected=[], submitted_ids=[], fills=[]
            )

        async with self._cycle_lock:
            try:
                from quant_platform.engines import engine_runner as facade

                await facade.hydrate_session_state(session)
                cycle_time = session.clock.now()
                effective_features = feature_data or await self._scheduled_feature_data(cycle_time)
                if self._config.uses_order_capable_external_broker and not effective_features:
                    log.error(
                        "engine_runner.feature_data_empty",
                        engine=self._config.engine_name,
                        mode=self._config.run_mode.value,
                        execution_backend=self._config.execution_backend.value,
                        detail="order-capable engine cycle requires nonempty feature data",
                    )
                validate_required_feature_schema(
                    engine_name=self._config.engine_name,
                    feature_data=effective_features,
                    required_features=self._config.required_features,
                    allow_empty=not self._config.uses_order_capable_external_broker,
                )
                effective_market_prices = await build_cycle_market_prices(
                    session=session,
                    instrument_contracts=self._config.instrument_contracts,
                    existing=market_prices,
                    as_of=cycle_time,
                )
                if self._config.uses_order_capable_external_broker:
                    validate_positive_feature_prices(
                        engine_name=self._config.engine_name,
                        feature_data=effective_features,
                        market_prices=effective_market_prices,
                    )

                if self._config.run_mode == RunMode.SHADOW:
                    result = await self._run_shadow_cycle(
                        effective_features,
                        effective_market_prices,
                    )
                    await self._run_shadow_boosting_cycle(
                        effective_features,
                        result.signals,
                        cycle_time,
                    )
                    await self._run_shadow_text_cycle(cycle_time, effective_market_prices)
                    return result

                if self._settings.v2.account_orchestrator_enabled:
                    result = await run_v2_proposal_cycle(
                        proposal_factory=lambda: self._generate_proposal_inner(
                            effective_features,
                            effective_market_prices,
                            cycle_time,
                        ),
                        event_bus=session.event_bus,
                        occurred_at=cycle_time,
                    )
                    self._result.cycles_completed += 1
                    return result

                result = await facade.run_strategy_cycle(
                    session=session,
                    feature_data=effective_features,
                    strategy_run=strategy_run,
                    market_prices=effective_market_prices,
                )

                self._result.cycles_completed += 1
                self._result.total_signals += len(result.signals)
                self._result.total_fills += len(result.fills)
                self._result.total_submitted += len(result.submitted_ids)
                self._result.total_rejected += len(result.rejected)
                await self._run_shadow_text_cycle(cycle_time, effective_market_prices)
                return result
            except Exception:
                from quant_platform.telemetry.metrics import record_cycle_error

                record_cycle_error(self._config.engine_name)
                raise


__all__ = ["EngineRunnerCycleMixin"]
