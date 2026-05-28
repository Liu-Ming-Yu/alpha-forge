"""Unit tests for StandardRiskPolicy.

Covers:
- evaluate(): gross exposure, single-name weight, sector weight, cash buffer
- check_order_limits(): single-name notional only (no cash-buffer duplication)
- Sell orders always pass check_order_limits
- Missing sector map skips sector check
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce
from quant_platform.core.domain.portfolio import PortfolioTarget, RiskLimits
from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.services.portfolio_service.risk_policy import StandardRiskPolicy

_UTC = UTC
_NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=_UTC)
_RUN = uuid.uuid4()
_TARGET = uuid.uuid4()
_INST_A = uuid.uuid4()
_INST_B = uuid.uuid4()
_INST_C = uuid.uuid4()


def _account(
    settled: Decimal = Decimal("100000"),
    nav: Decimal | None = None,
    positions: tuple[PositionSnapshot, ...] = (),
) -> AccountSnapshot:
    nav = nav or settled
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=settled,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=settled,
        net_asset_value=nav,
        positions=positions,
    )


def _limits(
    max_single_name: Decimal = Decimal("0.10"),
    max_sector: Decimal = Decimal("0.30"),
    max_gross: Decimal = Decimal("0.95"),
    min_cash: Decimal = Decimal("0.05"),
) -> RiskLimits:
    return RiskLimits(
        limits_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        effective_from=_NOW,
        max_single_name_weight=max_single_name,
        max_sector_weight=max_sector,
        max_gross_exposure=max_gross,
        max_daily_turnover=Decimal("0.20"),
        min_cash_buffer=min_cash,
        max_drawdown_halt=Decimal("-0.15"),
    )


def _target(
    weights: dict[uuid.UUID, Decimal],
    cash: Decimal = Decimal("0.10"),
) -> PortfolioTarget:
    return PortfolioTarget(
        target_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        as_of=_NOW,
        regime_id=uuid.uuid4(),
        weights=weights,
        cash_target_weight=cash,
    )


def _buy(instrument: uuid.UUID, qty: int, limit: Decimal) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET,
        instrument_id=instrument,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=limit,
    )


# ---------------------------------------------------------------------------
# evaluate() portfolio-level tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_all_checks_pass(self) -> None:
        policy = StandardRiskPolicy()
        target = _target({_INST_A: Decimal("0.08")}, cash=Decimal("0.92"))
        decision = policy.evaluate(target, _account(), _limits())
        assert decision.approved

    def test_gross_exposure_exceeded(self) -> None:
        policy = StandardRiskPolicy()
        target = _target(
            {_INST_A: Decimal("0.05"), _INST_B: Decimal("0.05")},
            cash=Decimal("0.90"),
        )
        decision = policy.evaluate(target, _account(), _limits(max_gross=Decimal("0.95")))
        assert decision.approved
        # Now exceed gross
        target2 = _target(
            {_INST_A: Decimal("0.50"), _INST_B: Decimal("0.50")},
            cash=Decimal("0"),
        )
        decision2 = policy.evaluate(target2, _account(), _limits(max_gross=Decimal("0.95")))
        assert not decision2.approved
        assert "gross exposure" in decision2.reason

    def test_single_name_weight_exceeded(self) -> None:
        policy = StandardRiskPolicy()
        target = _target({_INST_A: Decimal("0.15")}, cash=Decimal("0.85"))
        decision = policy.evaluate(target, _account(), _limits(max_single_name=Decimal("0.10")))
        assert not decision.approved
        assert "single-name" in decision.reason

    def test_sector_weight_exceeded(self) -> None:
        sector_map = {_INST_A: "Technology", _INST_B: "Technology"}
        policy = StandardRiskPolicy(sector_map=sector_map)
        target = _target(
            {_INST_A: Decimal("0.09"), _INST_B: Decimal("0.09")},
            cash=Decimal("0.82"),
        )
        decision = policy.evaluate(target, _account(), _limits(max_sector=Decimal("0.15")))
        assert not decision.approved
        assert "sector" in decision.reason.lower()

    def test_cash_buffer_too_low(self) -> None:
        policy = StandardRiskPolicy()
        target = _target({_INST_A: Decimal("0.09")}, cash=Decimal("0.01"))
        decision = policy.evaluate(target, _account(), _limits(min_cash=Decimal("0.05")))
        assert not decision.approved
        assert "cash" in decision.reason.lower()


# ---------------------------------------------------------------------------
# check_order_limits() per-order tests
# ---------------------------------------------------------------------------


class TestCheckOrderLimits:
    def test_within_single_name_limit(self) -> None:
        policy = StandardRiskPolicy()
        intent = _buy(_INST_A, 10, Decimal("50"))  # $500
        account = _account(nav=Decimal("100000"))
        decision = policy.check_order_limits(intent, account, _limits())
        assert decision.approved

    def test_exceeds_single_name_limit(self) -> None:
        policy = StandardRiskPolicy()
        # 10% of $100000 = $10000 max, order for $15000
        intent = _buy(_INST_A, 150, Decimal("100"))
        decision = policy.check_order_limits(intent, _account(nav=Decimal("100000")), _limits())
        assert not decision.approved
        assert "single-name" in decision.reason

    def test_no_cash_buffer_check(self) -> None:
        """check_order_limits must NOT reject on cash buffer grounds."""
        policy = StandardRiskPolicy()
        # notional = 1 * $5 = $5, NAV = $100, single-name limit = 10% = $10
        # Order is within single-name limit, but min_cash_buffer is set to 90%
        # which would fail if cash were checked — proving it is NOT checked.
        intent = _buy(_INST_A, 1, Decimal("5"))
        account = _account(settled=Decimal("100"), nav=Decimal("100"))
        limits = _limits(min_cash=Decimal("0.90"))
        decision = policy.check_order_limits(intent, account, limits)
        assert decision.approved  # cash buffer is NOT checked here

    def test_sell_always_passes(self) -> None:
        policy = StandardRiskPolicy()
        sell = OrderIntent(
            order_id=uuid.uuid4(),
            strategy_run_id=_RUN,
            portfolio_target_id=_TARGET,
            instrument_id=_INST_A,
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            created_at=_NOW,
        )
        decision = policy.check_order_limits(sell, _account(), _limits())
        assert decision.approved


# ---------------------------------------------------------------------------
# ETF correlation group tests (P1-2)
# ---------------------------------------------------------------------------


class TestETFCorrelationGroups:
    _SPY = uuid.uuid4()
    _QQQ = uuid.uuid4()
    _IWM = uuid.uuid4()

    def _target(self, spy_w: float, qqq_w: float, iwm_w: float = 0.0) -> PortfolioTarget:
        weights = {}
        if spy_w:
            weights[self._SPY] = Decimal(str(spy_w))
        if qqq_w:
            weights[self._QQQ] = Decimal(str(qqq_w))
        if iwm_w:
            weights[self._IWM] = Decimal(str(iwm_w))
        total = sum(weights.values(), Decimal("0"))
        cash = max(Decimal("0"), Decimal("1") - total)
        return PortfolioTarget(
            target_id=uuid.uuid4(),
            strategy_run_id=_RUN,
            regime_id=uuid.uuid4(),
            as_of=_NOW,
            weights=weights,
            cash_target_weight=cash,
        )

    def _policy(self) -> StandardRiskPolicy:
        return StandardRiskPolicy(
            etf_groups={"us_broad": {self._SPY, self._QQQ, self._IWM}},
            etf_group_cap_multiplier=1.5,
        )

    def _limits_5pct(self) -> RiskLimits:
        return _limits(max_single_name=Decimal("0.05"))

    def test_group_combined_weight_rejected(self) -> None:
        """SPY + QQQ each at 4% = 8% combined > 5% × 1.5 = 7.5% cap."""
        policy = self._policy()
        target = self._target(spy_w=0.04, qqq_w=0.04)
        decision = policy.evaluate(target, _account(), self._limits_5pct())
        assert not decision.approved
        assert "us_broad" in decision.reason
        assert "group cap" in decision.reason

    def test_group_combined_weight_allowed(self) -> None:
        """SPY at 3% + QQQ at 3% = 6% combined < 7.5% cap → passes."""
        policy = self._policy()
        target = self._target(spy_w=0.03, qqq_w=0.03)
        decision = policy.evaluate(target, _account(), self._limits_5pct())
        assert decision.approved

    def test_empty_groups_no_check(self) -> None:
        """No ETF groups configured → policy behaves identically to default."""
        policy = StandardRiskPolicy()  # no etf_groups
        target = self._target(spy_w=0.05, qqq_w=0.05)
        # gross = 0.10 + 0.90 cash, each 5% single-name = exactly at limit
        decision = policy.evaluate(target, _account(), self._limits_5pct())
        assert decision.approved  # no group check fires

    def test_single_instrument_in_group_behaves_normally(self) -> None:
        """A single instrument within the group doesn't trigger a group breach."""
        policy = self._policy()
        target = self._target(spy_w=0.05, qqq_w=0.0)  # only SPY in group
        decision = policy.evaluate(target, _account(), self._limits_5pct())
        assert decision.approved
