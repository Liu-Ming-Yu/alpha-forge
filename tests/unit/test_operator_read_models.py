"""Unit tests for operator read models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.application.operator_api.read_models import OperatorReadModelBuilder
from quant_platform.core.contracts import BrokerHealth, BrokerHealthStatus
from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.domain.production import EngineBudget, NavSnapshot, OrderAllocation
from quant_platform.core.events import (
    KillSwitchActivated,
    OrderApproved,
    OrderFilled,
    OrderRejected,
    ReconciliationCompleted,
)
from quant_platform.infrastructure.event_bus import InMemoryEventBus
from quant_platform.infrastructure.performance import InMemoryPerformanceRepository
from quant_platform.infrastructure.repositories import (
    InMemoryOrderRepository,
    InMemoryPositionRepository,
)
from quant_platform.infrastructure.repositories.multi_engine_governance import (
    InMemoryMultiEngineGovernanceRepository,
)
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.execution_service.orders.order_throttle import OrderThrottle
from quant_platform.services.portfolio_service.cash_ledger import CashLedger
from quant_platform.services.portfolio_service.settlement_calendar import SettlementCalendar

_UTC = UTC
_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=_UTC)


@pytest.mark.asyncio
async def test_paper_gate_metrics_projection() -> None:
    clock = FakeClock(_NOW)
    strategy_run_id = uuid.uuid4()
    order_id = uuid.uuid4()
    instrument_id = uuid.uuid4()

    snapshot = AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=Decimal("10000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("10000"),
        net_asset_value=Decimal("10000"),
        positions=(),
    )

    ledger = CashLedger(
        clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
    )
    throttle = OrderThrottle(clock)
    order_repo = InMemoryOrderRepository()
    position_repo = InMemoryPositionRepository()
    bus = InMemoryEventBus()

    intent = OrderIntent(
        order_id=order_id,
        strategy_run_id=strategy_run_id,
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=Decimal("100"),
    )
    await order_repo.save_intent(intent)

    await bus.publish(
        OrderApproved(
            event_id=uuid.uuid4(),
            occurred_at=_NOW,
            order_id=order_id,
            reservation_id=uuid.uuid4(),
        )
    )
    await bus.publish(
        OrderRejected(
            event_id=uuid.uuid4(),
            occurred_at=_NOW,
            order_id=order_id,
            reason="broker rejection: test",
        )
    )
    await bus.publish(
        OrderFilled(
            event_id=uuid.uuid4(),
            occurred_at=_NOW,
            order_id=order_id,
            fill_id=uuid.uuid4(),
            filled_quantity=10,
            fill_price=Decimal("101"),
            is_complete=True,
        )
    )
    await bus.publish(
        ReconciliationCompleted(
            event_id=uuid.uuid4(),
            occurred_at=_NOW,
            strategy_run_id=strategy_run_id,
            discrepancies_found=2,
            discrepancies_resolved=1,
            requires_operator_action=False,
        )
    )
    await bus.publish(
        KillSwitchActivated(
            event_id=uuid.uuid4(),
            occurred_at=_NOW,
            activated_by="strategy_cycle",
            reason="cash drift exceeded tolerance",
        )
    )

    builder = OperatorReadModelBuilder(
        clock=clock,
        cash_ledger=ledger,
        throttle=throttle,
        order_repo=order_repo,
        position_repo=position_repo,
        event_bus=bus,
    )

    metrics = await builder.paper_gate_metrics(strategy_run_id)
    assert metrics.orders_considered == 2
    assert metrics.reject_rate == Decimal("0.5")
    assert metrics.broker_error_rate == Decimal("1")
    assert metrics.reconcile_discrepancies == 2
    assert metrics.cash_drift_incidents == 1
    assert metrics.average_fill_slippage_bps is not None
    assert metrics.average_fill_slippage_bps == Decimal("100")


# ---------------------------------------------------------------------------
# R-OBS-05: broker_health() must reflect a real broker probe, not the kill
# switch state.
# ---------------------------------------------------------------------------


class _StubBrokerProbe:
    """Minimal BrokerSessionGateway stub used to drive broker_health()."""

    def __init__(self, status: BrokerHealthStatus, detail: str = "ok") -> None:
        self._status = status
        self._detail = detail

    async def health_check(self) -> BrokerHealth:
        return BrokerHealth(
            status=self._status,
            detail=self._detail,
            latency_ms=12,
            last_heartbeat_at=_NOW,
        )


def _simple_ledger(clock: FakeClock) -> CashLedger:
    return CashLedger(
        clock=clock,
        settlement_calendar=SettlementCalendar(),
        initial_snapshot=AccountSnapshot(
            snapshot_id=uuid.uuid4(),
            as_of=_NOW,
            settled_cash=Decimal("1000"),
            unsettled_cash=Decimal("0"),
            reserved_cash=Decimal("0"),
            available_cash=Decimal("1000"),
            net_asset_value=Decimal("1000"),
            positions=(),
        ),
    )


@pytest.mark.asyncio
async def test_broker_health_reports_connected_when_probe_ok_even_if_kill_switch_on() -> None:
    """Kill switch armed + broker actually connected → connected=True."""
    from quant_platform.core.contracts import BrokerHealthStatus

    clock = FakeClock(_NOW)
    throttle = OrderThrottle(clock)
    throttle.activate_kill_switch("operator halt", activated_by="reconciliation")

    builder = OperatorReadModelBuilder(
        clock=clock,
        cash_ledger=_simple_ledger(clock),
        throttle=throttle,
        order_repo=InMemoryOrderRepository(),
        position_repo=InMemoryPositionRepository(),
        event_bus=InMemoryEventBus(),
        account_broker=_StubBrokerProbe(BrokerHealthStatus.CONNECTED),
    )

    health = await builder.broker_health()
    assert health.connected is True, "connected must reflect the broker probe, not the kill switch"
    assert health.kill_switch_active is True
    assert health.status == "connected"


@pytest.mark.asyncio
async def test_broker_health_reports_disconnected_when_probe_times_out() -> None:
    """Broker down + kill switch off → connected=False and kill_switch_active=False."""
    from quant_platform.core.contracts import BrokerHealthStatus

    clock = FakeClock(_NOW)
    throttle = OrderThrottle(clock)

    class _TimingOutProbe:
        async def health_check(self):
            raise TimeoutError("probe timed out")

    builder = OperatorReadModelBuilder(
        clock=clock,
        cash_ledger=_simple_ledger(clock),
        throttle=throttle,
        order_repo=InMemoryOrderRepository(),
        position_repo=InMemoryPositionRepository(),
        event_bus=InMemoryEventBus(),
        account_broker=_TimingOutProbe(),
    )

    health = await builder.broker_health()
    assert health.connected is False
    assert health.kill_switch_active is False
    assert health.status == BrokerHealthStatus.DISCONNECTED.value
    assert "timed out" in health.detail


@pytest.mark.asyncio
async def test_broker_health_falls_back_to_throttle_without_probe() -> None:
    """No account_broker injected: throttle-derived view is retained."""
    clock = FakeClock(_NOW)
    throttle = OrderThrottle(clock)

    builder = OperatorReadModelBuilder(
        clock=clock,
        cash_ledger=_simple_ledger(clock),
        throttle=throttle,
        order_repo=InMemoryOrderRepository(),
        position_repo=InMemoryPositionRepository(),
        event_bus=InMemoryEventBus(),
    )

    health = await builder.broker_health()
    assert health.connected is True  # kill switch off means connected
    assert health.kill_switch_active is False


@pytest.mark.asyncio
async def test_strategy_lifecycle_uses_performance_repository() -> None:
    clock = FakeClock(_NOW)
    run_id = uuid.uuid4()
    performance = InMemoryPerformanceRepository()
    for idx, nav in enumerate([Decimal("1000"), Decimal("1100"), Decimal("1050")]):
        await performance.save_nav_snapshot(
            NavSnapshot(
                snapshot_id=uuid.uuid4(),
                strategy_run_id=run_id,
                as_of=_NOW.replace(day=_NOW.day + idx),
                net_asset_value=nav,
                gross_exposure=Decimal(str(idx)),
                cash=Decimal("100"),
            )
        )
    builder = OperatorReadModelBuilder(
        clock=clock,
        cash_ledger=_simple_ledger(clock),
        throttle=OrderThrottle(clock),
        order_repo=InMemoryOrderRepository(),
        position_repo=InMemoryPositionRepository(),
        performance_repo=performance,
        event_bus=InMemoryEventBus(),
    )

    view = await builder.current_strategy_lifecycle(
        run_id,
        engine_name="xsec",
        engine_version="1.0.0",
    )

    assert view.rolling_sharpe_90d != 0
    assert view.max_drawdown_realized < 0
    assert "turnover=" in view.recommendation


@pytest.mark.asyncio
async def test_multi_engine_budget_and_allocation_views() -> None:
    clock = FakeClock(_NOW)
    repo = InMemoryMultiEngineGovernanceRepository()
    await repo.save_engine_budget(
        EngineBudget(
            engine_name="cross_sectional_equity_v1",
            engine_version="0.1.0",
            run_mode="paper",
            capital_weight=Decimal("0.70"),
            max_gross=Decimal("0.70"),
            max_turnover=Decimal("0.20"),
        )
    )
    await repo.save_engine_budget(
        EngineBudget(
            engine_name="etf_macro_allocator_v1",
            engine_version="0.1.0",
            run_mode="paper",
            capital_weight=Decimal("0.25"),
            max_gross=Decimal("0.25"),
            max_turnover=Decimal("0.10"),
        )
    )
    order_id = uuid.uuid4()
    allocation = OrderAllocation(
        allocation_id=uuid.uuid4(),
        order_id=order_id,
        engine_name="etf_macro_allocator_v1",
        strategy_run_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        allocated_weight=Decimal("0.05"),
        allocated_notional=Decimal("5000"),
    )
    await repo.save_order_allocations([allocation])

    builder = OperatorReadModelBuilder(
        clock=clock,
        cash_ledger=_simple_ledger(clock),
        throttle=OrderThrottle(clock),
        order_repo=InMemoryOrderRepository(),
        position_repo=InMemoryPositionRepository(),
        event_bus=InMemoryEventBus(),
        multi_engine_repo=repo,
    )

    budgets = await builder.engine_budgets()
    exposure = await builder.combined_exposure()
    allocations = await builder.order_allocations(order_id)

    assert [budget.engine_name for budget in budgets] == [
        "cross_sectional_equity_v1",
        "etf_macro_allocator_v1",
    ]
    assert exposure.enabled_engines == 2
    assert exposure.allocated_capital_weight == Decimal("0.95")
    assert exposure.reserved_cash_weight == Decimal("0.05")
    assert allocations[0].engine_name == "etf_macro_allocator_v1"
    assert allocations[0].allocated_notional == Decimal("5000")
