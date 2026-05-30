"""Session and maintenance-scheduler wiring for engine runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.bootstrap.broker.live_broker_wiring import validate_ib_paper_execution
from quant_platform.bootstrap.data.feature_plugins import build_feature_registry
from quant_platform.bootstrap.session.public_api import (
    create_ib_paper_session,
    create_live_session,
    create_paper_session,
)
from quant_platform.engines.framework.types import EngineConfig, ExecutionBackend, RunMode
from quant_platform.engines.market_data.provider import build_account_market_data_provider
from quant_platform.engines.runtime.live import (
    assert_v2_is_only_live_submitter,
    bootstrap_ib_paper_snapshot,
    bootstrap_live_snapshot,
)
from quant_platform.services.data_service.maintenance.maintenance_scheduler import (
    DataMaintenanceScheduler,
)

if TYPE_CHECKING:
    from quant_platform.application.runtime.state import Session
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import MarketDataProvider, SignalModel
    from quant_platform.core.domain.research.runs import StrategyRun
    from quant_platform.services.portfolio_service.portfolio_constructor import (
        LongOnlyPortfolioConstructor,
    )


async def create_engine_runtime_session(
    *,
    config: EngineConfig,
    settings: PlatformSettings,
    strategy_run: StrategyRun,
    signal_model: SignalModel,
    portfolio_constructor: LongOnlyPortfolioConstructor,
) -> Session:
    """Create and connect the paper or live session for an engine."""
    if config.execution_backend == ExecutionBackend.IB_PAPER and config.run_mode != RunMode.PAPER:
        raise ValueError("execution_backend='ib-paper' is only valid with run_mode='paper'")

    if config.run_mode in (RunMode.SHADOW, RunMode.PAPER) and (
        config.execution_backend == ExecutionBackend.SIMULATED
    ):
        session = create_paper_session(
            settings=settings,
            initial_cash=config.initial_cash,
            strategy_run_id=strategy_run.run_id,
            signal_model=signal_model,
            portfolio_constructor=portfolio_constructor,
            instrument_contracts=config.instrument_contracts or None,
        )
        await session.broker.connect()
        return session

    if config.run_mode == RunMode.PAPER:
        if not config.instrument_contracts:
            raise ValueError(
                "EngineRunner IB paper mode requires instrument_contracts with con_id mappings"
            )
        validate_ib_paper_execution(settings, config.instrument_contracts)
        snapshot = await bootstrap_ib_paper_snapshot(
            broker_settings=settings.broker,
            instrument_contracts=config.instrument_contracts,
        )
        session = create_ib_paper_session(
            settings=settings,
            initial_snapshot=snapshot,
            strategy_run_id=strategy_run.run_id,
            signal_model=signal_model,
            portfolio_constructor=portfolio_constructor,
            instrument_contracts=config.instrument_contracts,
        )
        await session.broker.connect()
        return session

    if not config.instrument_contracts:
        raise ValueError(
            "EngineRunner LIVE mode requires instrument_contracts with con_id mappings"
        )
    assert_v2_is_only_live_submitter(settings)
    snapshot = await bootstrap_live_snapshot(
        broker_settings=settings.broker,
        instrument_contracts=config.instrument_contracts,
    )
    session = create_live_session(
        settings=settings,
        initial_snapshot=snapshot,
        strategy_run_id=strategy_run.run_id,
        signal_model=signal_model,
        portfolio_constructor=portfolio_constructor,
        instrument_contracts=config.instrument_contracts,
    )
    await session.broker.connect()
    return session


def build_engine_maintenance_scheduler(
    *,
    session: Session,
    settings: PlatformSettings,
    injected_market_data_provider: MarketDataProvider | None,
) -> DataMaintenanceScheduler:
    """Build the engine maintenance scheduler with the best available data provider."""
    market_data_provider = injected_market_data_provider
    if market_data_provider is None:
        market_data_provider = build_account_market_data_provider(
            session.account_broker,
            max_bar_age_minutes=(settings.data_ingest.max_bar_age_minutes or None),
            daily_max_bar_age_minutes=(settings.data_ingest.daily_max_bar_age_minutes or None),
        )
    # Opt-in vendor refresh: when a vendor chain is configured
    # (QP__DATA_INGEST__BAR_FETCH_FALLBACK_CHAIN), the maintenance loop pulls fresh
    # EOD bars for stale names each cycle — keeps a multi-day paper soak current
    # with no live IB feed. None when unconfigured ⇒ existing IB/provider behavior.
    from quant_platform.services.data_service.feeds.ingest_bar_fetcher_factory import (
        build_vendor_bar_fetcher,
    )

    bar_fetcher = build_vendor_bar_fetcher(settings, bar_seconds=86400)
    return DataMaintenanceScheduler(
        instruments=session.contract_master.list_active(),
        bar_store=session.bar_store,
        universe_manager=session.universe_manager,
        feature_repo=session.feature_repo,
        market_data_provider=market_data_provider,
        bar_fetcher=bar_fetcher,
        feature_registry=build_feature_registry(session.feature_repo),
    )
