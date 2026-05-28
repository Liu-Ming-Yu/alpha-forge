from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.orders import OrderSide
from quant_platform.core.domain.research import IntradayBacktestSpec
from quant_platform.services.research_service.backtesting.slippage import IBKRCommissionSchedule
from quant_platform.services.research_service.intraday.backtesting.helpers import (
    advance_settlements,
    apply_fill,
    assert_feature_availability,
    max_drawdown,
)
from quant_platform.services.research_service.intraday.backtesting.types import IntradayFillArtifact
from quant_platform.services.research_service.intraday.vectorized.execution import vectorized_fill


def test_assert_feature_availability_fails_closed_on_late_snapshot() -> None:
    decision_time = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    spec = IntradayBacktestSpec(
        strategy_name="availability",
        strategy_version="0.1.0",
        start=decision_time,
        end=decision_time + timedelta(minutes=1),
        initial_capital=Decimal("100000"),
        decision_times=(decision_time,),
        dataset_ids=(uuid.uuid4(),),
        universe_name="fixture",
        feature_set_version="fixture",
        model_version="fixture",
    )

    with pytest.raises(ValueError, match="look-ahead feature availability"):
        assert_feature_availability(
            spec,
            {decision_time: decision_time + timedelta(seconds=1)},
        )


def test_apply_fill_tracks_t_plus_one_sell_settlement() -> None:
    executed_at = datetime(2026, 1, 2, 15, 0, tzinfo=UTC)
    instrument_id = uuid.uuid4()
    fill = IntradayFillArtifact(
        order_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=OrderSide.SELL.value,
        tactic="test",
        executed_at=executed_at,
        quantity=5,
        requested_quantity=5,
        residual_quantity=0,
        arrival_price=Decimal("100"),
        decision_price=Decimal("100"),
        minute_vwap=Decimal("100"),
        fill_price=Decimal("100"),
        spread_bps=Decimal("0"),
        participation_rate=Decimal("0.01"),
        slippage_bps=Decimal("0"),
        implementation_shortfall_bps=Decimal("0"),
        commission=Decimal("1"),
        is_complete=True,
    )

    settled, unsettled, settlements = apply_fill(
        fill,
        Decimal("1000"),
        Decimal("0"),
        [],
        {instrument_id: 10},
        {instrument_id: Decimal("80")},
    )

    assert settled == Decimal("1000")
    assert unsettled == Decimal("499")
    settled, unsettled, settlements = advance_settlements(
        executed_at.date() + timedelta(days=1),
        settled,
        unsettled,
        settlements,
    )
    assert settled == Decimal("1499")
    assert unsettled == Decimal("0")
    assert settlements == []


def test_vectorized_fill_uses_decision_bar_and_commission() -> None:
    instrument_id = uuid.uuid4()
    decision_time = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    bar = MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=decision_time,
        bar_seconds=60,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=1000,
        vwap=Decimal("100.25"),
    )

    fill = vectorized_fill(
        strategy_run_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        decision_time=decision_time,
        quantity=10,
        reference_price=Decimal("100"),
        minute_bars=[bar],
        commission_schedule=IBKRCommissionSchedule(),
    )

    assert fill.minute_vwap == Decimal("100.25")
    assert fill.commission > 0
    assert fill.implementation_shortfall_bps >= 0
    assert fill.is_complete is True


def test_max_drawdown_reports_worst_peak_to_trough_loss() -> None:
    assert max_drawdown([Decimal("100"), Decimal("110"), Decimal("99")]) == Decimal("-0.1")
