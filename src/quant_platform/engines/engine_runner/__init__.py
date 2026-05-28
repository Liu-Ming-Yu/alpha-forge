"""Strategy engine runner facade for shadow, paper, and live modes."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Protocol

import structlog

from quant_platform.config import PlatformSettings
from quant_platform.engines.engine_runner.cycle import EngineRunnerCycleMixin
from quant_platform.engines.engine_runner.proposals import EngineRunnerProposalMixin
from quant_platform.engines.engine_runner.shadow import EngineRunnerShadowMixin
from quant_platform.engines.framework.initialization_wiring import (
    build_engine_portfolio_constructor,
    build_signal_model,
    build_strategy_run,
)
from quant_platform.engines.framework.model_registry_wiring import (
    register_engine_model_and_schedule_job,
)
from quant_platform.engines.framework.types import (
    EngineConfig,
    EngineRunResult,
    ExecutionBackend,
    RunMode,
)
from quant_platform.engines.runtime.live import (
    assert_v2_is_only_live_submitter,
    bootstrap_live_snapshot,
)
from quant_platform.engines.session.public_api import (
    hydrate_session_state,
    model_registry_preflight,
    run_strategy_cycle,
)
from quant_platform.engines.shadow.scorer_wiring import (
    build_shadow_boosting_scorer,
    build_shadow_text_scorer,
)
from quant_platform.infrastructure.postgres.model_registry import (
    PostgresModelRegistry,
    build_model_registry,
)
from quant_platform.services.research_service.features.pipeline.feature_pipeline import (
    FEATURE_SET_VERSION,
)

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from quant_platform.application.runtime.state import Session
    from quant_platform.core.contracts import MarketDataProvider, SignalModel
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.research import StrategyRun
    from quant_platform.engines.shadow.boosting_cycle import ShadowBoostingScorer
    from quant_platform.engines.shadow.text_cycle import ShadowTextCycleScorer
    from quant_platform.services.data_service.maintenance.maintenance_scheduler import (
        DataMaintenanceScheduler,
    )
    from quant_platform.services.portfolio_service.portfolio_constructor import (
        LongOnlyPortfolioConstructor,
    )
    from quant_platform.services.research_service.modeling.registry.model_registry import (
        FeatureJob,
        InMemoryModelRegistry,
    )


class EngineSessionFactory(Protocol):
    """Builds and connects the runtime Session for an engine.

    Injected by the bootstrap composition layer so the engine run loop never
    imports session composition code (keeps the composition tier acyclic).
    """

    async def __call__(
        self,
        *,
        config: EngineConfig,
        settings: PlatformSettings,
        strategy_run: StrategyRun,
        signal_model: SignalModel,
        portfolio_constructor: LongOnlyPortfolioConstructor,
    ) -> Session: ...


class EngineSchedulerFactory(Protocol):
    """Builds the data-maintenance scheduler for an engine. Injected by bootstrap."""

    def __call__(
        self,
        *,
        session: Session,
        settings: PlatformSettings,
        injected_market_data_provider: MarketDataProvider | None,
    ) -> DataMaintenanceScheduler: ...


class EngineRunner(EngineRunnerShadowMixin, EngineRunnerProposalMixin, EngineRunnerCycleMixin):
    """Runs one strategy engine through initialization and cycle dispatch."""

    def __init__(
        self,
        config: EngineConfig,
        settings: PlatformSettings | None = None,
        *,
        market_data_provider: MarketDataProvider | None = None,
    ) -> None:
        self._config = config
        self._settings = settings or PlatformSettings()
        self._injected_market_data_provider = market_data_provider
        self._session: Session | None = None
        self._strategy_run: StrategyRun | None = None
        self._maintenance_scheduler: DataMaintenanceScheduler | None = None
        self._model_registry: InMemoryModelRegistry | PostgresModelRegistry = build_model_registry(
            self._settings.storage.postgres_dsn
        )
        self._feature_jobs: list[FeatureJob] = []
        self._shadow_text_scorer: ShadowTextCycleScorer | None = None
        self._shadow_boosting_scorer: ShadowBoostingScorer | None = None
        self._cycle_lock = asyncio.Lock()
        self._result = EngineRunResult(
            run_id=uuid.uuid4(),
            engine_name=config.engine_name,
            run_mode=config.run_mode,
        )

    def _portfolio_constructor(self) -> LongOnlyPortfolioConstructor:
        """Build the engine-scoped constructor honoring ``EngineConfig``."""
        return build_engine_portfolio_constructor(
            config=self._config,
            settings=self._settings,
        )

    async def initialize(
        self,
        *,
        session_factory: EngineSessionFactory,
        scheduler_factory: EngineSchedulerFactory,
    ) -> None:
        """Create the runtime session, model registration, and shadow dependencies.

        ``session_factory`` and ``scheduler_factory`` are injected by the
        bootstrap composition layer; the engine run loop never imports session
        composition code, which keeps the composition tier acyclic.
        """
        from quant_platform.infrastructure.support.clock import WallClock

        if self._settings.storage.postgres_dsn:
            from quant_platform.infrastructure.support.migrations import verify_alembic_head

            await verify_alembic_head(self._settings.storage.postgres_dsn)

        clock = WallClock()
        self._strategy_run = build_strategy_run(config=self._config, clock=clock)
        signal_model = build_signal_model(self._config)

        self._session = await session_factory(
            config=self._config,
            settings=self._settings,
            strategy_run=self._strategy_run,
            signal_model=signal_model,
            portfolio_constructor=self._portfolio_constructor(),
        )

        if self._session is None:
            raise RuntimeError("engine runtime session was not initialized")
        if self._config.uses_order_capable_external_broker:
            await model_registry_preflight(
                self._session,
                strategy_name=self._config.engine_name,
                engine_version=self._config.engine_version,
                require_match=True,
            )
        else:
            await model_registry_preflight(
                self._session,
                strategy_name=self._config.engine_name,
                engine_version=self._config.engine_version,
            )
        scheduled_job = await register_engine_model_and_schedule_job(
            self._model_registry,
            engine_name=self._config.engine_name,
            engine_version=self._config.engine_version,
            feature_set_version=FEATURE_SET_VERSION,
            run_mode=self._config.run_mode,
            max_positions=self._config.max_positions,
            interval_seconds=self._config.rebalance_interval_seconds,
            as_of=clock.now(),
            max_model_age_hours=self._settings.risk.max_model_age_hours,
        )
        self._feature_jobs = [scheduled_job]
        self._maintenance_scheduler = scheduler_factory(
            session=self._session,
            settings=self._settings,
            injected_market_data_provider=self._injected_market_data_provider,
        )
        self._shadow_text_scorer = build_shadow_text_scorer(
            settings=self._settings,
            session=self._session,
        )
        self._shadow_boosting_scorer = build_shadow_boosting_scorer(
            settings=self._settings,
            run_mode=self._config.run_mode,
            session=self._session,
        )

        log.info(
            "engine_runner.initialized",
            engine=self._config.engine_name,
            mode=self._config.run_mode.value,
            execution_backend=self._config.execution_backend.value,
            cash=str(self._config.initial_cash),
        )

    def _assert_v2_is_only_live_submitter(self) -> None:
        assert_v2_is_only_live_submitter(self._settings)

    async def _bootstrap_live_snapshot(
        self,
        instrument_contracts: dict[uuid.UUID, dict[str, object]],
    ) -> AccountSnapshot:
        return await bootstrap_live_snapshot(
            broker_settings=self._settings.broker,
            instrument_contracts=instrument_contracts,
        )

    async def shutdown(self) -> EngineRunResult:
        """Disconnect broker and return aggregated results."""
        if self._session is not None:
            await self._session.broker.disconnect()
        return self._result


__all__ = [
    "EngineConfig",
    "EngineRunner",
    "ExecutionBackend",
    "RunMode",
    "hydrate_session_state",
    "run_strategy_cycle",
]
