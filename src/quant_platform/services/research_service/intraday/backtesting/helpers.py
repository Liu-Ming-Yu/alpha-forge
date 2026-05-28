"""Shared helpers for intraday backtest engines."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.orders import (
    ExecutionTactic,
    OrderIntent,
    OrderSide,
    OrderType,
    VenueRoute,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.core.domain.research import IntradayBacktestSpec
    from quant_platform.services.research_service.intraday.backtesting.types import (
        IntradayFillArtifact,
    )


def assert_feature_availability(
    spec: IntradayBacktestSpec,
    feature_available_at: Mapping[datetime, datetime],
) -> None:
    for decision_time in spec.decision_times:
        available_at = feature_available_at.get(decision_time)
        if available_at is None:
            raise ValueError(f"missing feature available_at for {decision_time.isoformat()}")
        if available_at > decision_time:
            raise ValueError(
                f"look-ahead feature availability: decision={decision_time.isoformat()} "
                f"available_at={available_at.isoformat()}"
            )


def route_execution_tactic(policy: object, intent: OrderIntent) -> VenueRoute:
    """Map an intent to a deterministic backtest execution tactic."""
    if intent.order_type == OrderType.MOC:
        tactic = ExecutionTactic.CLOSE_AUCTION_MOC
        urgency = Decimal("1")
    elif intent.order_type == OrderType.LOC:
        tactic = ExecutionTactic.CLOSE_AUCTION_LOC
        urgency = Decimal("0.75")
    elif (
        bool(getattr(policy, "passive_limit_enabled", False))
        and intent.order_type == OrderType.LIMIT
    ):
        tactic = ExecutionTactic.PASSIVE_LIMIT
        urgency = Decimal("0.25")
    else:
        tactic = ExecutionTactic.URGENCY_LIMIT
        urgency = Decimal("0.50")
    max_participation = Decimal(str(getattr(policy, "max_adv_participation_pct", "0.05")))
    return VenueRoute(
        route_id=uuid.uuid4(),
        venue="IBKR_SMART",
        tactic=tactic,
        max_participation_rate=max_participation,
        urgency=urgency,
    )


def prices_at(
    minute_bars: Mapping[uuid.UUID, list[MarketBar]],
    as_of: datetime,
) -> dict[uuid.UUID, Decimal]:
    prices: dict[uuid.UUID, Decimal] = {}
    for instrument_id, bars in minute_bars.items():
        past = [bar for bar in bars if bar.timestamp <= as_of]
        if past:
            prices[instrument_id] = sorted(past, key=lambda bar: bar.timestamp)[-1].close
    return prices


def bar_at_or_before(bars: list[MarketBar], as_of: datetime) -> MarketBar | None:
    past = [bar for bar in bars if bar.timestamp <= as_of]
    if past:
        return sorted(past, key=lambda bar: bar.timestamp)[-1]
    return None


def bars_for_order_window(bars: list[MarketBar], decision_time: datetime) -> list[MarketBar]:
    return [
        bar
        for bar in sorted(bars, key=lambda item: item.timestamp)
        if bar.timestamp >= decision_time and bar.timestamp.date() == decision_time.date()
    ]


def account_snapshot(
    *,
    as_of: datetime,
    settled_cash: Decimal,
    unsettled_cash: Decimal,
    positions: Mapping[uuid.UUID, int],
    avg_cost: Mapping[uuid.UUID, Decimal],
    minute_bars: Mapping[uuid.UUID, list[MarketBar]],
) -> AccountSnapshot:
    snapshots: list[PositionSnapshot] = []
    market_prices = prices_at(minute_bars, as_of)
    for instrument_id, quantity in positions.items():
        if quantity <= 0:
            continue
        price = market_prices.get(instrument_id, avg_cost.get(instrument_id, Decimal("1")))
        cost = avg_cost.get(instrument_id, price)
        market_value = Decimal(quantity) * price
        snapshots.append(
            PositionSnapshot(
                snapshot_id=uuid.uuid4(),
                instrument_id=instrument_id,
                quantity=quantity,
                average_cost=cost,
                market_price=price,
                market_value=market_value,
                unrealised_pnl=market_value - Decimal(quantity) * cost,
                as_of=as_of,
                source="intraday_backtest",
            )
        )
    nav = settled_cash + unsettled_cash + sum(pos.market_value for pos in snapshots)
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=as_of,
        settled_cash=settled_cash,
        unsettled_cash=unsettled_cash,
        reserved_cash=Decimal("0"),
        available_cash=settled_cash,
        net_asset_value=nav,
        positions=tuple(snapshots),
        source="intraday_backtest",
    )


def apply_fill(
    fill: IntradayFillArtifact,
    settled_cash: Decimal,
    unsettled_cash: Decimal,
    settlements: list[tuple[date, Decimal]],
    positions: dict[uuid.UUID, int],
    avg_cost: dict[uuid.UUID, Decimal],
) -> tuple[Decimal, Decimal, list[tuple[date, Decimal]]]:
    notional = fill.fill_price * Decimal(fill.quantity)
    if fill.side == OrderSide.BUY.value:
        settled_cash -= notional + fill.commission
        old_qty = positions.get(fill.instrument_id, 0)
        old_cost = avg_cost.get(fill.instrument_id, fill.fill_price)
        new_qty = old_qty + fill.quantity
        avg_cost[fill.instrument_id] = (
            ((Decimal(old_qty) * old_cost) + notional) / Decimal(new_qty)
            if new_qty > 0
            else fill.fill_price
        )
        positions[fill.instrument_id] = new_qty
    else:
        sell_qty = min(positions.get(fill.instrument_id, 0), fill.quantity)
        positions[fill.instrument_id] = max(0, positions.get(fill.instrument_id, 0) - sell_qty)
        proceeds = notional - fill.commission
        unsettled_cash += proceeds
        settlements.append((fill.executed_at.date() + timedelta(days=1), proceeds))
    return settled_cash, unsettled_cash, settlements


def advance_settlements(
    today: date,
    settled_cash: Decimal,
    unsettled_cash: Decimal,
    settlements: list[tuple[date, Decimal]],
) -> tuple[Decimal, Decimal, list[tuple[date, Decimal]]]:
    remaining: list[tuple[date, Decimal]] = []
    for settle_date, amount in settlements:
        if settle_date <= today:
            settled_cash += amount
            unsettled_cash -= amount
        else:
            remaining.append((settle_date, amount))
    return settled_cash, max(Decimal("0"), unsettled_cash), remaining


def max_drawdown(nav_curve: list[Decimal]) -> Decimal:
    peak = nav_curve[0] if nav_curve else Decimal("0")
    worst = Decimal("0")
    for nav in nav_curve:
        if nav > peak:
            peak = nav
        if peak > 0:
            worst = min(worst, (nav - peak) / peak)
    return worst
