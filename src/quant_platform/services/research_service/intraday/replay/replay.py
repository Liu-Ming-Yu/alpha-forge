"""Intraday order tactic replay model."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.orders import (
    ExecutionTactic,
    OrderIntent,
    OrderSide,
    OrderType,
)
from quant_platform.services.research_service.backtesting.slippage import IBKRCommissionSchedule
from quant_platform.services.research_service.intraday.backtesting.types import (
    IntradayFillArtifact,
    IntradayReplayOrderResult,
)

if TYPE_CHECKING:
    from quant_platform.core.domain.market_data import MarketBar


class IntradayTacticReplayModel:
    """Replay approved orders against 1-minute OHLCV bars."""

    def __init__(
        self,
        *,
        commission_schedule: IBKRCommissionSchedule | None = None,
        stale_price_bps: Decimal = Decimal("1.0"),
    ) -> None:
        self._commission = commission_schedule or IBKRCommissionSchedule()
        self._stale_price_bps = stale_price_bps

    def replay_order(
        self,
        intent: OrderIntent,
        bars: list[MarketBar],
        *,
        tactic: ExecutionTactic,
        max_participation_rate: Decimal,
        decision_price: Decimal,
    ) -> IntradayReplayOrderResult:
        """Return deterministic partial fills for one order."""
        if not bars:
            return IntradayReplayOrderResult(intent, tactic, (), intent.quantity, comparable=False)
        remaining = intent.quantity
        fills: list[IntradayFillArtifact] = []
        replay_bars = sorted(bars, key=lambda bar: bar.timestamp)
        if tactic in {ExecutionTactic.CLOSE_AUCTION_MOC, ExecutionTactic.CLOSE_AUCTION_LOC}:
            replay_bars = replay_bars[-1:]

        for bar in replay_bars:
            if remaining <= 0:
                break
            if not self._bar_crosses_order(intent, bar, tactic):
                continue
            rate = self._effective_participation_rate(tactic, max_participation_rate)
            capacity = int(Decimal(bar.volume) * rate)
            if capacity <= 0:
                continue
            quantity = min(remaining, capacity)
            if quantity <= 0:
                continue
            fill_price = self._fill_price(intent, bar, quantity, rate, tactic)
            spread_bps = Decimal(str(self._spread_bps(fill_price, tactic)))
            participation = Decimal(quantity) / Decimal(max(1, bar.volume))
            shortfall = self._shortfall_bps(intent.side, decision_price, fill_price)
            commission = self._commission.compute(quantity, fill_price)
            remaining -= quantity
            fills.append(
                IntradayFillArtifact(
                    order_id=intent.order_id,
                    instrument_id=intent.instrument_id,
                    side=intent.side.value,
                    tactic=tactic.value,
                    executed_at=bar.timestamp,
                    quantity=quantity,
                    requested_quantity=intent.quantity,
                    residual_quantity=remaining,
                    arrival_price=decision_price,
                    decision_price=decision_price,
                    minute_vwap=bar.vwap or bar.close,
                    fill_price=fill_price,
                    spread_bps=spread_bps,
                    participation_rate=participation,
                    slippage_bps=shortfall,
                    implementation_shortfall_bps=shortfall,
                    commission=commission,
                    is_complete=remaining == 0,
                )
            )

        return IntradayReplayOrderResult(
            intent=intent,
            tactic=tactic,
            fills=tuple(fills),
            residual_quantity=remaining,
            comparable=remaining == 0,
        )

    @staticmethod
    def _bar_crosses_order(
        intent: OrderIntent,
        bar: MarketBar,
        tactic: ExecutionTactic,
    ) -> bool:
        if intent.order_type in {OrderType.MARKET, OrderType.MOC}:
            return True
        if tactic == ExecutionTactic.URGENCY_LIMIT:
            return True
        limit = intent.limit_price or bar.close
        if intent.side == OrderSide.BUY:
            return bar.low <= limit
        return bar.high >= limit

    @staticmethod
    def _effective_participation_rate(
        tactic: ExecutionTactic,
        max_participation_rate: Decimal,
    ) -> Decimal:
        if tactic == ExecutionTactic.PASSIVE_LIMIT:
            return max(Decimal("0.0001"), max_participation_rate * Decimal("0.50"))
        if tactic in {ExecutionTactic.CLOSE_AUCTION_MOC, ExecutionTactic.CLOSE_AUCTION_LOC}:
            return max_participation_rate
        return max(Decimal("0.0001"), max_participation_rate * Decimal("0.75"))

    def _fill_price(
        self,
        intent: OrderIntent,
        bar: MarketBar,
        quantity: int,
        participation_rate: Decimal,
        tactic: ExecutionTactic,
    ) -> Decimal:
        reference = bar.vwap or bar.close
        spread_bps = Decimal(str(self._spread_bps(reference, tactic)))
        impact_bps = Decimal(str(100.0 * 0.6 * math.sqrt(float(participation_rate))))
        adverse_bps = (spread_bps / Decimal("2")) + impact_bps + self._stale_price_bps
        if tactic in {ExecutionTactic.CLOSE_AUCTION_MOC, ExecutionTactic.CLOSE_AUCTION_LOC}:
            adverse_bps *= Decimal("1.25")
        bump = adverse_bps / Decimal("10000")
        del quantity
        if intent.side == OrderSide.BUY:
            return max(reference * (Decimal("1") + bump), Decimal("0.01"))
        return max(reference * (Decimal("1") - bump), Decimal("0.01"))

    @staticmethod
    def _spread_bps(price: Decimal, tactic: ExecutionTactic) -> float:
        p = float(price)
        if p < 5:
            base = 20.0
        elif p < 20:
            base = 8.0
        elif p < 100:
            base = 4.0
        else:
            base = 2.0
        if tactic == ExecutionTactic.PASSIVE_LIMIT:
            return base * 0.6
        if tactic in {ExecutionTactic.CLOSE_AUCTION_MOC, ExecutionTactic.CLOSE_AUCTION_LOC}:
            return base * 1.5
        return base

    @staticmethod
    def _shortfall_bps(side: OrderSide, decision_price: Decimal, fill_price: Decimal) -> Decimal:
        if decision_price <= 0:
            return Decimal("0")
        if side == OrderSide.BUY:
            return ((fill_price - decision_price) / decision_price) * Decimal("10000")
        return ((decision_price - fill_price) / decision_price) * Decimal("10000")
