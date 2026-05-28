"""Public paper/live session factory implementations."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from quant_platform.bootstrap.broker.live_broker_wiring import (
    build_ib_paper_broker_gateways,
    build_live_broker_gateways,
)
from quant_platform.bootstrap.session.defaults import (
    assert_live_session_defaults,
    log_session_defaults,
    regime_thresholds_from_settings,
)
from quant_platform.bootstrap.session.factory import build_session
from quant_platform.config import PlatformSettings
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.engines.account.orchestrator import AccountExecutionOrchestrator
from quant_platform.engines.multi_engine import MultiEngineRunner
from quant_platform.infrastructure.support.clock import WallClock
from quant_platform.infrastructure.v2.postgres import build_v2_repository_bundle
from quant_platform.services.execution_service.orders.router import DefaultExecutionRouter
from quant_platform.services.execution_service.simulated_broker import SimulatedBrokerGateway
from quant_platform.services.portfolio_service.optimizer import ConstraintAwareOptimizer
from quant_platform.services.signal_service.regime_detector import MarketRegimeDetector

if TYPE_CHECKING:
    from quant_platform.application.runtime.state import Session
    from quant_platform.core.contracts import Clock, LifecycleFeed, SignalModel
    from quant_platform.services.portfolio_service.portfolio_constructor import (
        LongOnlyPortfolioConstructor,
        SimpleRegimeDetector,
    )


def create_paper_session_impl(
    settings: PlatformSettings | None = None,
    *,
    initial_cash: Decimal = Decimal("50000"),
    strategy_run_id: uuid.UUID | None = None,
    clock: Clock | None = None,
    signal_model: SignalModel | None = None,
    portfolio_constructor: LongOnlyPortfolioConstructor | None = None,
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
    regime_detector: SimpleRegimeDetector | MarketRegimeDetector | None = None,
) -> Session:
    """Build a paper-trading session backed by SimulatedBrokerGateway."""
    settings = settings or PlatformSettings()
    run_id = strategy_run_id or uuid.uuid4()
    clk = clock or WallClock()
    now = clk.now()

    broker = SimulatedBrokerGateway(clock=clk, initial_cash=initial_cash)
    snapshot = AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=now,
        settled_cash=initial_cash,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=initial_cash,
        net_asset_value=initial_cash,
        positions=(),
    )

    session = build_session(
        settings,
        clk,
        broker,
        snapshot,
        run_id,
        signal_model=signal_model,
        portfolio_constructor=portfolio_constructor,
        instrument_contracts=instrument_contracts,
        regime_detector=regime_detector,
    )
    session = maybe_attach_v2_orchestrator(session, settings)
    log_session_defaults(session, mode="paper")
    return session


def create_live_session_impl(
    settings: PlatformSettings | None = None,
    *,
    initial_snapshot: AccountSnapshot,
    strategy_run_id: uuid.UUID | None = None,
    signal_model: SignalModel | None = None,
    portfolio_constructor: LongOnlyPortfolioConstructor | None = None,
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
    regime_detector: SimpleRegimeDetector | MarketRegimeDetector | None = None,
) -> Session:
    """Build a live-trading session backed by the configured broker gateway."""
    settings = settings or PlatformSettings()
    run_id = strategy_run_id or uuid.uuid4()
    clock = WallClock()

    contracts = instrument_contracts or {}
    trading_broker, account_broker = build_live_broker_gateways(
        settings=settings,
        initial_snapshot=initial_snapshot,
        instrument_contracts=contracts,
        clock=clock,
    )

    live_regime = regime_detector
    if live_regime is None and settings.regime.enabled:
        live_regime = MarketRegimeDetector(
            thresholds=regime_thresholds_from_settings(settings),
            disagree_haircut=settings.regime.disagree_confidence_haircut,
        )

    session = build_session(
        settings,
        clock,
        trading_broker,
        initial_snapshot,
        run_id,
        signal_model=signal_model,
        portfolio_constructor=portfolio_constructor,
        account_broker=account_broker,
        trading_broker=trading_broker,
        lifecycle_feed=cast("LifecycleFeed", trading_broker),
        instrument_contracts=contracts,
        regime_detector=live_regime,
    )
    if hasattr(trading_broker, "set_execution_policy"):
        trading_broker.set_execution_policy(session.execution_policy)
    session = maybe_attach_v2_orchestrator(session, settings)
    log_session_defaults(session, mode="live")
    assert_live_session_defaults(session)
    return session


def create_ib_paper_session_impl(
    settings: PlatformSettings | None = None,
    *,
    initial_snapshot: AccountSnapshot,
    strategy_run_id: uuid.UUID | None = None,
    signal_model: SignalModel | None = None,
    portfolio_constructor: LongOnlyPortfolioConstructor | None = None,
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
    regime_detector: SimpleRegimeDetector | MarketRegimeDetector | None = None,
) -> Session:
    """Build a paper-trading session backed by paper TWS/Gateway."""
    settings = settings or PlatformSettings()
    run_id = strategy_run_id or uuid.uuid4()
    clock = WallClock()
    contracts = instrument_contracts or {}
    trading_broker, account_broker = build_ib_paper_broker_gateways(
        settings=settings,
        initial_snapshot=initial_snapshot,
        instrument_contracts=contracts,
        clock=clock,
    )

    paper_regime = regime_detector
    if paper_regime is None and settings.regime.enabled:
        paper_regime = MarketRegimeDetector(
            thresholds=regime_thresholds_from_settings(settings),
            disagree_haircut=settings.regime.disagree_confidence_haircut,
        )

    session = build_session(
        settings,
        clock,
        trading_broker,
        initial_snapshot,
        run_id,
        signal_model=signal_model,
        portfolio_constructor=portfolio_constructor,
        account_broker=account_broker,
        trading_broker=trading_broker,
        lifecycle_feed=cast("LifecycleFeed", trading_broker),
        instrument_contracts=contracts,
        regime_detector=paper_regime,
    )
    if hasattr(trading_broker, "set_execution_policy"):
        trading_broker.set_execution_policy(session.execution_policy)
    session = maybe_attach_v2_orchestrator(session, settings)
    log_session_defaults(session, mode="ib_paper")
    return session


def maybe_attach_v2_orchestrator(
    session: Session,
    settings: PlatformSettings,
) -> Session:
    """Build and attach AccountExecutionOrchestrator when V2 is enabled."""
    if not (settings.v2.enabled and settings.v2.account_orchestrator_enabled):
        return session

    v2_bundle = build_v2_repository_bundle(settings)
    order_planner = session.order_planner
    if order_planner is None:
        raise RuntimeError("V2 orchestrator requires a portfolio target order planner")
    optimizer = ConstraintAwareOptimizer()
    router = DefaultExecutionRouter(
        policy=session.execution_tactic_policy,
        broker=session.trading_broker,
    )
    stub_multi_engine = MultiEngineRunner(settings=settings, budgets=())
    orchestrator = AccountExecutionOrchestrator(
        multi_engine=stub_multi_engine,
        optimizer=optimizer,
        order_planner=order_planner,
        approve_ctrl=session.approve_ctrl,
        submit_ctrl=session.submit_ctrl,
        order_state=v2_bundle.order_state,
        execution_router=router,
        risk_repo=v2_bundle.risk_models,
    )
    session.account_orchestrator = orchestrator
    session.v2_order_state = v2_bundle.order_state
    session.v2_risk_repo = v2_bundle.risk_models
    return session
