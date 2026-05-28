"""Per-cycle execution quality extraction for backtest replay."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from ..artifacts.backtest_artifacts import (
    BacktestFillArtifact,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable
    from datetime import datetime

    from quant_platform.core.contracts import (
        BacktestCycleResult,
        BacktestReplayBroker,
        BacktestSession,
    )


@dataclass(frozen=True)
class CycleExecutionStats:
    """Execution-quality facts extracted from one simulated cycle."""

    fill_artifacts: list[BacktestFillArtifact]
    commission: Decimal
    slippage_cost: Decimal
    slippage_bps: float
    filled_quantity: int
    requested_quantity: int
    participation: list[float]
    shortfall_bps: list[float]


async def collect_cycle_execution_stats(
    *,
    result: BacktestCycleResult,
    session: BacktestSession,
    broker: BacktestReplayBroker,
    cycle_ts: datetime,
    slippage_bps_from_prices: Callable[..., float],
) -> CycleExecutionStats:
    """Build fill artifacts and aggregate execution stats for one cycle."""

    fill_artifacts: list[BacktestFillArtifact] = []
    cycle_commission = Decimal("0")
    cycle_slippage_cost = Decimal("0")
    cycle_slippage_bps = 0.0
    submitted_order_ids = set(result.submitted_ids)
    requested_by_order: dict[uuid.UUID, int] = {
        intent.order_id: intent.quantity
        for intent in result.approved
        if intent.order_id in submitted_order_ids
    }
    cycle_requested_quantity = sum(requested_by_order.values())
    cycle_filled_quantity = 0
    cycle_participation: list[float] = []
    cycle_shortfall_bps: list[float] = []

    for fill in result.fills:
        intent = await session.order_repo.get_intent(fill.order_id)
        model_price = (
            intent.limit_price
            if intent is not None and intent.limit_price is not None
            else fill.fill_price
        )
        commission = fill.commission
        cycle_commission += commission

        slippage = slippage_bps_from_prices(
            side=fill.side.value,
            model_price=model_price,
            fill_price=fill.fill_price,
        )
        cycle_slippage_bps += slippage

        slippage_notional = abs(fill.fill_price - model_price) * Decimal(str(fill.quantity))
        cycle_slippage_cost += slippage_notional
        cycle_filled_quantity += fill.quantity

        plan = broker.execution_plan_for(fill.order_id)
        requested_quantity = (
            plan.requested_quantity
            if plan is not None
            else requested_by_order.get(fill.order_id, fill.quantity)
        )
        filled_quantity = plan.filled_quantity if plan is not None else fill.quantity
        fill_ratio = filled_quantity / requested_quantity if requested_quantity > 0 else 0.0
        adv_shares = plan.adv_shares_20d if plan is not None else 0.0
        participation_pct = plan.participation_pct if plan is not None else 0.0
        spread_bps = plan.spread_bps if plan is not None else 0.0
        implementation_shortfall_bps = (
            plan.implementation_shortfall_bps if plan is not None else slippage
        )
        is_complete = plan.is_complete if plan is not None else True
        cycle_participation.append(participation_pct)
        cycle_shortfall_bps.append(implementation_shortfall_bps)

        fill_artifacts.append(
            BacktestFillArtifact(
                cycle_ts=cycle_ts,
                order_id=fill.order_id,
                instrument_id=fill.instrument_id,
                side=fill.side.value,
                quantity=fill.quantity,
                requested_quantity=requested_quantity,
                filled_quantity=filled_quantity,
                fill_ratio=fill_ratio,
                raw_fill_price=model_price,
                adjusted_fill_price=fill.fill_price,
                commission=commission,
                adv_shares_20d=adv_shares,
                participation_pct=participation_pct,
                spread_bps=spread_bps,
                slippage_bps=slippage,
                slippage_cost=slippage_notional,
                implementation_shortfall_bps=implementation_shortfall_bps,
                is_complete=is_complete,
            )
        )

    return CycleExecutionStats(
        fill_artifacts=fill_artifacts,
        commission=cycle_commission,
        slippage_cost=cycle_slippage_cost,
        slippage_bps=cycle_slippage_bps,
        filled_quantity=cycle_filled_quantity,
        requested_quantity=cycle_requested_quantity,
        participation=cycle_participation,
        shortfall_bps=cycle_shortfall_bps,
    )
