"""Per-cycle evidence aggregation for the simple parity backtest."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.services.research_service.backtesting.simple.backtest_cycle_execution import (
    collect_cycle_execution_stats,
)

from .backtest_artifacts import (
    BacktestCycleMetrics,
    BacktestFillArtifact,
)

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.core.contracts import (
        BacktestCycleResult,
        BacktestReplayBroker,
        BacktestSession,
    )

SlippageBpsCalculator = Callable[..., float]


@dataclass(frozen=True)
class SimpleBacktestCycleEvidence:
    nav: Decimal
    metrics: BacktestCycleMetrics
    fill_artifacts: list[BacktestFillArtifact]
    commission: Decimal
    slippage_cost: Decimal
    slippage_bps: float


async def collect_simple_backtest_cycle_evidence(
    *,
    ts: datetime,
    result: BacktestCycleResult,
    session: BacktestSession,
    broker: BacktestReplayBroker,
    slippage_bps_from_prices: SlippageBpsCalculator,
) -> SimpleBacktestCycleEvidence:
    """Build cycle metrics and fill artifacts from a live-like cycle result."""
    execution = await collect_cycle_execution_stats(
        result=result,
        session=session,
        broker=broker,
        cycle_ts=ts,
        slippage_bps_from_prices=slippage_bps_from_prices,
    )
    account = await broker.sync_account()
    nav = account.net_asset_value
    metrics = BacktestCycleMetrics(
        timestamp=ts,
        nav=nav,
        total_commission=execution.commission,
        total_slippage_bps=execution.slippage_bps,
        signals_count=len(result.signals),
        fills_count=len(result.fills),
        orders_count=len(result.submitted_ids),
        fill_rate=(
            execution.filled_quantity / execution.requested_quantity
            if execution.requested_quantity > 0
            else 0.0
        ),
        average_participation_pct=(
            sum(execution.participation) / len(execution.participation)
            if execution.participation
            else 0.0
        ),
        implementation_shortfall_bps=(
            sum(execution.shortfall_bps) / len(execution.shortfall_bps)
            if execution.shortfall_bps
            else 0.0
        ),
    )
    return SimpleBacktestCycleEvidence(
        nav=nav,
        metrics=metrics,
        fill_artifacts=execution.fill_artifacts,
        commission=execution.commission,
        slippage_cost=execution.slippage_cost,
        slippage_bps=execution.slippage_bps,
    )
