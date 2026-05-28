from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quant_platform.services.research_service.backtesting.artifacts.backtest_artifacts import (
    BacktestFillArtifact,
)
from quant_platform.services.research_service.backtesting.simple.backtest_performance import (
    gross_turnover_from_fills,
    max_drawdown_from_nav,
    periods_per_year_from_rebalances,
)


def _fill(*, quantity: int, price: Decimal) -> BacktestFillArtifact:
    return BacktestFillArtifact(
        cycle_ts=datetime(2026, 1, 2, tzinfo=UTC),
        order_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side="buy",
        quantity=quantity,
        requested_quantity=quantity,
        filled_quantity=quantity,
        fill_ratio=1.0,
        raw_fill_price=price,
        adjusted_fill_price=price,
        commission=Decimal("0"),
        adv_shares_20d=1_000_000.0,
        participation_pct=0.01,
        spread_bps=1.0,
        slippage_bps=0.0,
        slippage_cost=Decimal("0"),
        implementation_shortfall_bps=0.0,
        is_complete=True,
    )


def test_gross_turnover_from_fills_uses_absolute_adjusted_notional() -> None:
    turnover = gross_turnover_from_fills(
        [_fill(quantity=10, price=Decimal("100")), _fill(quantity=5, price=Decimal("50"))],
        Decimal("10000"),
    )

    assert turnover == Decimal("0.1250")


def test_max_drawdown_from_nav_tracks_worst_peak_to_trough() -> None:
    drawdown = max_drawdown_from_nav(
        [Decimal("100"), Decimal("125"), Decimal("100"), Decimal("130")]
    )

    assert drawdown == Decimal("-0.2")


def test_periods_per_year_from_rebalances_uses_observed_spacing() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    assert periods_per_year_from_rebalances([start, start + timedelta(days=7)]) == 365.0 / 7.0
    assert periods_per_year_from_rebalances([start]) == 252.0
