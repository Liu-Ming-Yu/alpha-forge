"""Vectorized intraday comparator fill construction."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.orders import ExecutionTactic, OrderSide
from quant_platform.services.research_service.intraday.backtesting.helpers import bar_at_or_before
from quant_platform.services.research_service.intraday.backtesting.types import IntradayFillArtifact
from quant_platform.services.research_service.intraday.replay.replay import (
    IntradayTacticReplayModel,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.services.research_service.backtesting.slippage import IBKRCommissionSchedule


def rebalance_to_target_percent(
    *,
    strategy_run_id: uuid.UUID,
    decision_time: datetime,
    target_weights: Mapping[uuid.UUID, Decimal],
    nav: Decimal,
    market_prices: Mapping[uuid.UUID, Decimal],
    minute_bars: Mapping[uuid.UUID, list[MarketBar]],
    positions: Mapping[uuid.UUID, int],
    settled_cash: Decimal,
    commission_schedule: IBKRCommissionSchedule,
) -> list[IntradayFillArtifact]:
    fills: list[IntradayFillArtifact] = []
    cash_remaining = settled_cash
    instrument_ids = sorted(set(positions) | set(target_weights), key=str)
    for instrument_id in instrument_ids:
        reference = market_prices.get(instrument_id)
        if reference is None or reference <= 0:
            continue
        current_qty = positions.get(instrument_id, 0)
        desired_value = nav * target_weights.get(instrument_id, Decimal("0"))
        desired_qty = int(desired_value / reference)
        delta = desired_qty - current_qty
        if delta == 0:
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        quantity = abs(delta)
        fill = vectorized_fill(
            strategy_run_id=strategy_run_id,
            instrument_id=instrument_id,
            side=side,
            decision_time=decision_time,
            quantity=quantity,
            reference_price=reference,
            minute_bars=minute_bars.get(instrument_id, []),
            commission_schedule=commission_schedule,
        )
        if side == OrderSide.BUY:
            total_cost = fill.fill_price * Decimal(fill.quantity) + fill.commission
            if total_cost > cash_remaining:
                affordable = int(cash_remaining / fill.fill_price)
                if affordable <= 0:
                    continue
                fill = vectorized_fill(
                    strategy_run_id=strategy_run_id,
                    instrument_id=instrument_id,
                    side=side,
                    decision_time=decision_time,
                    quantity=affordable,
                    reference_price=reference,
                    minute_bars=minute_bars.get(instrument_id, []),
                    commission_schedule=commission_schedule,
                )
                total_cost = fill.fill_price * Decimal(fill.quantity) + fill.commission
            cash_remaining -= total_cost
        fills.append(fill)
    return fills


def vectorized_fill(
    *,
    strategy_run_id: uuid.UUID,
    instrument_id: uuid.UUID,
    side: OrderSide,
    decision_time: datetime,
    quantity: int,
    reference_price: Decimal,
    minute_bars: list[MarketBar],
    commission_schedule: IBKRCommissionSchedule,
) -> IntradayFillArtifact:
    bar = bar_at_or_before(minute_bars, decision_time)
    minute_vwap = (bar.vwap or bar.close) if bar is not None else reference_price
    spread_bps = Decimal(
        str(IntradayTacticReplayModel._spread_bps(reference_price, ExecutionTactic.URGENCY_LIMIT))
    )
    bump = (spread_bps / Decimal("2")) / Decimal("10000")
    fill_price = (
        reference_price * (Decimal("1") + bump)
        if side == OrderSide.BUY
        else reference_price * (Decimal("1") - bump)
    )
    participation = Decimal("0")
    if bar is not None and bar.volume > 0:
        participation = Decimal(quantity) / Decimal(bar.volume)
    shortfall = IntradayTacticReplayModel._shortfall_bps(side, reference_price, fill_price)
    return IntradayFillArtifact(
        order_id=uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"vectorized:{strategy_run_id}:{instrument_id}:{decision_time.isoformat()}:{side.value}:{quantity}",
        ),
        instrument_id=instrument_id,
        side=side.value,
        tactic="vectorized_target_percent",
        executed_at=decision_time,
        quantity=quantity,
        requested_quantity=quantity,
        residual_quantity=0,
        arrival_price=reference_price,
        decision_price=reference_price,
        minute_vwap=minute_vwap,
        fill_price=fill_price,
        spread_bps=spread_bps,
        participation_rate=participation,
        slippage_bps=shortfall,
        implementation_shortfall_bps=shortfall,
        commission=commission_schedule.compute(quantity, fill_price),
        is_complete=True,
    )
