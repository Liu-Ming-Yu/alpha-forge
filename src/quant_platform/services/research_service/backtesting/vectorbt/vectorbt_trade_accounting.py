"""Trade accounting helpers for VectorBT portfolio simulation."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from ..slippage import (
    IBKRCommissionSchedule,
    SlippageModel,
    SlippageSide,
)
from .vectorbt_trade_fills import (
    build_vectorbt_fill_artifact,
    participation_pct_for_shares,
)
from .vectorbt_trade_types import (
    CycleTradeResult,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from ..simple.backtest_execution_model import (
        BacktestExecutionModel,
    )


def exit_non_target_positions(
    *,
    ts: datetime,
    positions: dict[uuid.UUID, Decimal],
    target_ids: list[uuid.UUID],
    prices: dict[uuid.UUID, Decimal],
    cash: Decimal,
    execution_model: BacktestExecutionModel,
    slippage_model: SlippageModel,
    commission_schedule: IBKRCommissionSchedule,
) -> CycleTradeResult:
    result = CycleTradeResult(cash=cash)
    for instrument_id in sorted(set(positions) - set(target_ids), key=str):
        shares = positions.pop(instrument_id)
        if shares <= 0:
            continue
        price = prices.get(instrument_id)
        if price is None or price <= 0:
            continue
        result = sell_position(
            result=result,
            ts=ts,
            instrument_id=instrument_id,
            shares=shares,
            price=price,
            execution_model=execution_model,
            slippage_model=slippage_model,
            commission_schedule=commission_schedule,
        )
    return result


def rebalance_target_positions(
    *,
    ts: datetime,
    positions: dict[uuid.UUID, Decimal],
    target_ids: list[uuid.UUID],
    prices: dict[uuid.UUID, Decimal],
    cash: Decimal,
    regime_scale: Decimal,
    execution_model: BacktestExecutionModel,
    slippage_model: SlippageModel,
    commission_schedule: IBKRCommissionSchedule,
) -> CycleTradeResult:
    result = CycleTradeResult(cash=cash)
    available_nav = cash + sum(
        positions.get(instrument_id, Decimal("0")) * prices.get(instrument_id, Decimal("0"))
        for instrument_id in positions
    )
    target_per_position = (available_nav * regime_scale) / Decimal(str(len(target_ids)))

    for instrument_id in target_ids:
        price = prices[instrument_id]
        target_shares = Decimal(str(int(float(target_per_position) / float(price))))
        current_shares = positions.get(instrument_id, Decimal("0"))
        delta = target_shares - current_shares
        if abs(delta) < 1:
            continue

        result = trade_to_delta(
            result=result,
            ts=ts,
            instrument_id=instrument_id,
            positions=positions,
            current_shares=current_shares,
            delta=delta,
            price=price,
            execution_model=execution_model,
            slippage_model=slippage_model,
            commission_schedule=commission_schedule,
        )
    return result


def sell_position(
    *,
    result: CycleTradeResult,
    ts: datetime,
    instrument_id: uuid.UUID,
    shares: Decimal,
    price: Decimal,
    execution_model: BacktestExecutionModel,
    slippage_model: SlippageModel,
    commission_schedule: IBKRCommissionSchedule,
) -> CycleTradeResult:
    adv_shares, spread_bps = execution_model.lookup_liquidity_params(instrument_id, price)
    slippage_bps = slippage_model.estimate_slippage(
        int(shares), adv_shares, spread_bps, SlippageSide.SELL
    )
    slippage_frac = Decimal(str(slippage_bps / 10_000.0))
    commission_usd = commission_schedule.compute(int(shares), price)
    result.cash += shares * price * (1 - slippage_frac) - commission_usd
    participation_pct = participation_pct_for_shares(int(shares), adv_shares)
    result.add_fill(
        build_vectorbt_fill_artifact(
            ts=ts,
            instrument_id=instrument_id,
            side="sell",
            shares=int(shares),
            price=price,
            slippage_frac=slippage_frac,
            commission=commission_usd,
            adv_shares=adv_shares,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
        ),
        commission=commission_usd,
        slippage_bps=slippage_bps,
        participation_pct=participation_pct,
    )
    return result


def trade_to_delta(
    *,
    result: CycleTradeResult,
    ts: datetime,
    instrument_id: uuid.UUID,
    positions: dict[uuid.UUID, Decimal],
    current_shares: Decimal,
    delta: Decimal,
    price: Decimal,
    execution_model: BacktestExecutionModel,
    slippage_model: SlippageModel,
    commission_schedule: IBKRCommissionSchedule,
) -> CycleTradeResult:
    adv_shares, spread_bps = execution_model.lookup_liquidity_params(instrument_id, price)
    shares_abs = int(abs(delta))
    commission_usd = commission_schedule.compute(shares_abs, price)

    if delta > 0:
        slippage_bps = slippage_model.estimate_slippage(
            shares_abs, adv_shares, spread_bps, SlippageSide.BUY
        )
        slippage_frac = Decimal(str(slippage_bps / 10_000.0))
        cost = delta * price * (1 + slippage_frac) + commission_usd
        if result.cash < cost:
            denom = float(price) * (1 + float(slippage_frac))
            affordable = max(0, int((float(result.cash) - float(commission_usd)) / denom))
            delta = Decimal(str(affordable)) - current_shares
            if delta <= 0:
                return result
            shares_abs = int(delta)
            commission_usd = commission_schedule.compute(shares_abs, price)
            cost = delta * price * (1 + slippage_frac) + commission_usd
        result.cash -= cost
        side = "buy"
    else:
        slippage_bps = slippage_model.estimate_slippage(
            shares_abs, adv_shares, spread_bps, SlippageSide.SELL
        )
        slippage_frac = Decimal(str(slippage_bps / 10_000.0))
        result.cash += abs(delta) * price * (1 - slippage_frac) - commission_usd
        side = "sell"

    positions[instrument_id] = current_shares + delta
    if positions[instrument_id] <= 0:
        positions.pop(instrument_id, None)

    participation_pct = participation_pct_for_shares(shares_abs, adv_shares)
    result.add_fill(
        build_vectorbt_fill_artifact(
            ts=ts,
            instrument_id=instrument_id,
            side=side,
            shares=shares_abs,
            price=price,
            slippage_frac=slippage_frac,
            commission=commission_usd,
            adv_shares=adv_shares,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
        ),
        commission=commission_usd,
        slippage_bps=slippage_bps,
        participation_pct=participation_pct,
    )
    return result
