"""Unit tests for PreTradeGate.

Covers:
- PASS: all three checks approve
- FAIL_CASH: cash gate rejects → risk and execution not evaluated
- FAIL_RISK: cash passes but risk rejects → execution not evaluated
- FAIL_EXECUTION_POLICY: cash and risk pass but kill switch is active
- GateDecision.passed property
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce
from quant_platform.core.domain.portfolio import RiskLimits
from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.services.execution_service.orders.order_throttle import OrderThrottle
from quant_platform.services.portfolio_service.cash_ledger import CashLedger
from quant_platform.services.portfolio_service.pretrade_gate import GateOutcome, PreTradeGate
from quant_platform.services.portfolio_service.risk_policy import StandardRiskPolicy
from quant_platform.services.portfolio_service.settlement_calendar import SettlementCalendar

_UTC = UTC
_NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=_UTC)
_RUN = uuid.uuid4()
_TARGET = uuid.uuid4()
_INSTRUMENT = uuid.uuid4()
_LIMITS_ID = uuid.uuid4()


class _FixedClock:
    def now(self) -> datetime:
        return _NOW

    def today(self) -> date:
        return _NOW.date()


def _account(settled: Decimal = Decimal("20000"), nav: Decimal | None = None) -> AccountSnapshot:
    nav = nav or settled
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=settled,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=settled,
        net_asset_value=nav,
        positions=(),
    )


def _limits(
    max_single_name: Decimal = Decimal("0.10"),
    min_cash_buffer: Decimal = Decimal("0.05"),
) -> RiskLimits:
    return RiskLimits(
        limits_id=_LIMITS_ID,
        strategy_run_id=_RUN,
        effective_from=_NOW,
        max_single_name_weight=max_single_name,
        max_sector_weight=Decimal("0.30"),
        max_gross_exposure=Decimal("0.95"),
        max_daily_turnover=Decimal("0.20"),
        min_cash_buffer=min_cash_buffer,
        max_drawdown_halt=Decimal("-0.15"),
    )


def _buy_intent(qty: int, limit_price: Decimal) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET,
        instrument_id=_INSTRUMENT,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=limit_price,
    )


@pytest.fixture
def gate() -> PreTradeGate:
    clock = _FixedClock()
    ledger = CashLedger(
        clock=clock,
        settlement_calendar=SettlementCalendar(),
        initial_snapshot=_account(),
    )
    risk = StandardRiskPolicy()
    throttle = OrderThrottle(clock, capacity=10, refill_rate=2.0)
    return PreTradeGate(cash_engine=ledger, risk_policy=risk, execution_policy=throttle)


class TestPreTradeGate:
    def test_pass_all_checks(self, gate: PreTradeGate) -> None:
        intent = _buy_intent(10, limit_price=Decimal("50"))  # $500 → 2.5% of $20000 NAV
        decision = gate.evaluate(intent, _account(), _limits())
        assert decision.passed
        assert decision.outcome == GateOutcome.PASS

    def test_fail_cash(self) -> None:
        """No settled cash → cash gate must fail."""
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(settled=Decimal("0")),
        )
        gate = PreTradeGate(
            cash_engine=ledger,
            risk_policy=StandardRiskPolicy(),
            execution_policy=OrderThrottle(clock),
        )
        intent = _buy_intent(10, limit_price=Decimal("50"))
        decision = gate.evaluate(intent, _account(settled=Decimal("0")), _limits())
        assert not decision.passed
        assert decision.outcome == GateOutcome.FAIL_CASH
        assert not decision.risk_decision.approved  # not evaluated
        assert not decision.execution_decision.approved  # not evaluated

    def test_fail_risk_single_name_limit(self) -> None:
        """Order notional exceeds single-name limit → risk gate must fail."""
        # NAV = $20000, max_single_name_weight = 1% → max notional = $200
        # Order: 50 shares × $50 = $2500 → exceeds $200
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(),  # $20000 settled
        )
        gate = PreTradeGate(
            cash_engine=ledger,
            risk_policy=StandardRiskPolicy(),
            execution_policy=OrderThrottle(clock),
        )
        tight_limits = _limits(max_single_name=Decimal("0.01"))  # 1% limit
        intent = _buy_intent(50, limit_price=Decimal("50"))  # $2500 notional
        decision = gate.evaluate(intent, _account(), tight_limits)
        assert not decision.passed
        assert decision.outcome == GateOutcome.FAIL_RISK
        assert decision.cash_decision.approved  # cash passed
        assert not decision.execution_decision.approved  # not evaluated

    def test_fail_execution_kill_switch(self) -> None:
        """Kill switch active → execution policy rejects after cash and risk pass."""
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(),
        )
        throttle = OrderThrottle(clock)
        throttle.activate_kill_switch("test", activated_by="unit_test")
        gate = PreTradeGate(
            cash_engine=ledger,
            risk_policy=StandardRiskPolicy(),
            execution_policy=throttle,
        )
        intent = _buy_intent(10, limit_price=Decimal("50"))
        decision = gate.evaluate(intent, _account(), _limits())
        assert not decision.passed
        assert decision.outcome == GateOutcome.FAIL_EXECUTION_POLICY
        assert decision.cash_decision.approved
        assert decision.risk_decision.approved

    def test_sell_passes_gate(self, gate: PreTradeGate) -> None:
        sell = OrderIntent(
            order_id=uuid.uuid4(),
            strategy_run_id=_RUN,
            portfolio_target_id=_TARGET,
            instrument_id=_INSTRUMENT,
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            created_at=_NOW,
        )
        pos = PositionSnapshot(
            snapshot_id=uuid.uuid4(),
            instrument_id=_INSTRUMENT,
            quantity=20,
            average_cost=Decimal("50"),
            market_price=Decimal("50"),
            market_value=Decimal("1000"),
            unrealised_pnl=Decimal("0"),
            as_of=_NOW,
        )
        account = AccountSnapshot(
            snapshot_id=uuid.uuid4(),
            as_of=_NOW,
            settled_cash=Decimal("20000"),
            unsettled_cash=Decimal("0"),
            reserved_cash=Decimal("0"),
            available_cash=Decimal("20000"),
            net_asset_value=Decimal("20000"),
            positions=(pos,),
        )
        decision = gate.evaluate(sell, account, _limits())
        assert decision.passed
