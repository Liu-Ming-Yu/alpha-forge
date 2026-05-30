"""Shared session object-graph composition."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from quant_platform.application.runtime.state import Session, SessionDrawdownGuard
from quant_platform.bootstrap.persistence.execution_stores import (
    build_completed_order_hint_store,
    build_kill_switch_store,
    build_pending_settlement_store,
)
from quant_platform.bootstrap.persistence.runtime_repositories import build_runtime_repositories
from quant_platform.bootstrap.session.components import (
    build_contract_master,
    build_default_portfolio_constructor,
    build_liquidity_checker,
    seed_universe_liquidity,
)
from quant_platform.bootstrap.session.defaults import (
    regime_thresholds_from_settings,
    risk_limits_from_settings,
)
from quant_platform.bootstrap.session.preflight import run_sector_mapping_preflight
from quant_platform.bootstrap.signal_models import build_default_primary_signal_model
from quant_platform.core.domain.orders import OrderType
from quant_platform.core.domain.production import ExecutionTacticPolicy
from quant_platform.services.data_service.reference.universe_manager import UniverseManager
from quant_platform.services.execution_service.account.account_state_coordinator import (
    AccountStateCoordinator,
)
from quant_platform.services.execution_service.orders.controllers import (
    ReconcileBrokerStateControllerImpl,
    SubmitOrdersControllerImpl,
)
from quant_platform.services.execution_service.orders.order_throttle import OrderThrottle
from quant_platform.services.execution_service.reconciliation import (
    ReconciliationEngine,
)
from quant_platform.services.execution_service.session.session_supervisor import (
    BrokerSessionSupervisor,
)
from quant_platform.services.governance_service.llm_live_startup import (
    assert_llm_live_startup_allowed,
)
from quant_platform.services.portfolio_service.cash_ledger import CashLedger
from quant_platform.services.portfolio_service.controllers import (
    ApproveOrdersControllerImpl,
    BuildPortfolioControllerImpl,
)
from quant_platform.services.portfolio_service.order_planner import (
    PortfolioTargetOrderPlanner,
)
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
    SimpleRegimeDetector,
)
from quant_platform.services.portfolio_service.pretrade_gate import PreTradeGate
from quant_platform.services.portfolio_service.risk_policy import StandardRiskPolicy
from quant_platform.services.portfolio_service.settlement_calendar import (
    SettlementCalendar,
)
from quant_platform.services.signal_service.controllers import (
    GenerateSignalsControllerImpl,
)
from quant_platform.services.signal_service.regime_detector import MarketRegimeDetector

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        BrokerGateway,
        BrokerOrderRoutingGateway,
        BrokerSessionGateway,
        Clock,
        LifecycleFeed,
        PredictionEvidenceRepository,
        SignalModel,
    )
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot


def build_session(
    settings: PlatformSettings,
    clock: Clock,
    broker: BrokerGateway,
    initial_snapshot: AccountSnapshot,
    strategy_run_id: uuid.UUID,
    signal_model: SignalModel | None = None,
    portfolio_constructor: LongOnlyPortfolioConstructor | None = None,
    account_broker: BrokerSessionGateway | None = None,
    trading_broker: BrokerOrderRoutingGateway | None = None,
    lifecycle_feed: LifecycleFeed | None = None,
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
    regime_detector: SimpleRegimeDetector | MarketRegimeDetector | None = None,
) -> Session:
    """Build the broker-agnostic runtime session graph."""
    now = clock.now()
    assert_llm_live_startup_allowed(settings, now=now)
    limits = risk_limits_from_settings(settings, strategy_run_id, now)
    contract_master = build_contract_master(instrument_contracts)
    universe_manager = UniverseManager(contract_master, settings.liquidity)
    seed_universe_liquidity(universe_manager, instrument_contracts, now)
    sector_map = universe_manager.sector_map()
    run_sector_mapping_preflight(contract_master, sector_map, settings)

    ledger = CashLedger(
        clock=clock,
        settlement_calendar=SettlementCalendar(),
        initial_snapshot=initial_snapshot,
        settings=settings.cash,
    )

    etf_uuid_groups: dict[str, set[uuid.UUID]] | None = None
    if settings.risk.etf_correlation_groups:
        symbol_to_uuid = {
            instr.symbol: instr.instrument_id
            for instr in contract_master.list_active()
            if hasattr(instr, "symbol") and instr.symbol
        }
        etf_uuid_groups = {}
        for group_name, symbols in settings.risk.etf_correlation_groups.items():
            ids = {symbol_to_uuid[s] for s in symbols if s in symbol_to_uuid}
            if ids:
                etf_uuid_groups[group_name] = ids

    risk = StandardRiskPolicy(
        sector_map=sector_map,
        etf_groups=etf_uuid_groups,
        etf_group_cap_multiplier=settings.risk.etf_group_cap_multiplier,
    )

    trading_calendar = None
    if settings.execution.trading_hours_enforced:
        from quant_platform.services.execution_service.support.trading_calendar import (
            DefaultTradingCalendar,
        )

        trading_calendar = DefaultTradingCalendar()

    kill_switch_store = build_kill_switch_store(settings.storage.postgres_dsn)
    throttle = OrderThrottle(
        clock,
        settings=settings.throttle,
        trading_calendar=trading_calendar,
        trading_hours_enforced=settings.execution.trading_hours_enforced,
        kill_switch_store=kill_switch_store,
    )
    gate = PreTradeGate(
        cash_engine=ledger,
        risk_policy=risk,
        execution_policy=throttle,
        liquidity_checker=build_liquidity_checker(
            universe_manager,
            allow_missing_profile=settings.liquidity.allow_missing_profile,
        ),
    )

    repositories = build_runtime_repositories(settings)
    approve_ctrl = ApproveOrdersControllerImpl(
        gate=gate,
        cash_engine=ledger,
        order_repo=repositories.order_repo,
        event_bus=repositories.event_bus,
        limits=limits,
    )

    routing_broker = trading_broker or broker
    session_broker = account_broker or broker
    lifecycle = lifecycle_feed or (
        cast("LifecycleFeed", broker) if hasattr(broker, "drain_lifecycle_events") else None
    )

    submit_ctrl = SubmitOrdersControllerImpl(
        broker=routing_broker,
        execution_policy=throttle,
        cash_engine=ledger,
        event_bus=repositories.event_bus,
        order_repo=repositories.order_repo,
        risk_policy=risk,
        limits=limits,
    )
    recon_engine = ReconciliationEngine(
        broker=session_broker,
        position_repo=repositories.position_repo,
        audit_sink=repositories.audit_sink,
        clock=clock,
        auto_correct_threshold=settings.risk.auto_correct_threshold,
    )
    recon_ctrl = ReconcileBrokerStateControllerImpl(
        engine=recon_engine,
        event_bus=repositories.event_bus,
        execution_policy=throttle,
    )

    coordinator = AccountStateCoordinator(
        cash_engine=ledger,
        event_bus=repositories.event_bus,
        clock=clock,
        strategy_run_id=strategy_run_id,
        order_repo=repositories.order_repo,
        pending_settlement_store=build_pending_settlement_store(settings.storage.postgres_dsn),
        completed_order_hint_store=build_completed_order_hint_store(settings.storage.postgres_dsn),
        execution_policy=throttle,
    )

    active_signal_model = signal_model or build_default_primary_signal_model(settings)
    signal_ctrl = GenerateSignalsControllerImpl(
        active_signal_model,
        repositories.event_bus,
        signal_contribution_repo=repositories.signal_contribution_repo,
        prediction_evidence_repo=cast(
            "PredictionEvidenceRepository",
            repositories.performance_repo,
        ),
    )

    constructor = portfolio_constructor or build_default_portfolio_constructor(settings)
    portfolio_ctrl = BuildPortfolioControllerImpl(constructor, risk, repositories.event_bus)

    if regime_detector is not None:
        active_regime_detector = regime_detector
    elif settings.regime.enabled:
        active_regime_detector = MarketRegimeDetector(
            thresholds=regime_thresholds_from_settings(settings),
            disagree_haircut=settings.regime.disagree_confidence_haircut,
        )
    else:
        active_regime_detector = SimpleRegimeDetector()

    supervisor = BrokerSessionSupervisor(
        session_gateway=session_broker,
        order_gateway=routing_broker,
        event_bus=repositories.event_bus,
        execution_policy=throttle,
        reconcile_controller=recon_ctrl,
        order_repo=repositories.order_repo,
        clock=clock,
        settings=settings.broker,
    )
    tactic_policy = ExecutionTacticPolicy(
        passive_limit_enabled=settings.execution.passive_limit_enabled,
        reprice_interval_seconds=settings.execution.reprice_interval_seconds,
        max_reprices_per_order=settings.execution.max_reprices_per_order,
        min_reprice_improvement_bps=settings.execution.min_reprice_improvement_bps,
        adverse_drift_escalate_bps=settings.execution.adverse_drift_escalate_bps,
        close_auction_enabled=settings.execution.close_auction_enabled,
        max_adv_participation_pct=settings.liquidity.adv_participation_pct,
        order_timeout_seconds=settings.execution.order_timeout_seconds,
    )
    order_type = OrderType.MOC if tactic_policy.close_auction_enabled else OrderType.LIMIT
    order_planner = PortfolioTargetOrderPlanner(
        clock,
        order_type=order_type,
        rebalance_threshold=settings.execution.rebalance_threshold,
    )

    return Session(
        settings=settings,
        clock=clock,
        broker=broker,
        account_broker=session_broker,
        trading_broker=routing_broker,
        lifecycle_feed=lifecycle,
        cash_engine=ledger,
        risk_policy=risk,
        execution_policy=throttle,
        pretrade_gate=gate,
        approve_ctrl=approve_ctrl,
        submit_ctrl=submit_ctrl,
        recon_engine=recon_engine,
        recon_ctrl=recon_ctrl,
        coordinator=coordinator,
        event_bus=repositories.event_bus,
        audit_sink=repositories.audit_sink,
        order_repo=repositories.order_repo,
        position_repo=repositories.position_repo,
        performance_repo=repositories.performance_repo,
        feature_repo=repositories.feature_repo,
        signal_contribution_repo=repositories.signal_contribution_repo,
        text_event_store=repositories.text_event_store,
        bar_store=repositories.bar_store,
        contract_master=contract_master,
        universe_manager=universe_manager,
        risk_limits=limits,
        supervisor=supervisor,
        signal_ctrl=signal_ctrl,
        portfolio_constructor=constructor,
        portfolio_ctrl=portfolio_ctrl,
        order_planner=order_planner,
        regime_detector=active_regime_detector,
        kill_switch_store=kill_switch_store,
        drawdown_guard=SessionDrawdownGuard(limits.max_drawdown_halt),
        execution_tactic_policy=tactic_policy,
    )
