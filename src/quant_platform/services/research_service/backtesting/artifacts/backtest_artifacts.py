"""Shared backtest artifact DTOs and metrics helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


@dataclass(frozen=True)
class BacktestCycleMetrics:
    """Per-cycle performance metrics collected during a backtest run."""

    timestamp: datetime
    nav: Decimal
    total_commission: Decimal
    total_slippage_bps: float
    signals_count: int
    fills_count: int
    orders_count: int
    fill_rate: float
    average_participation_pct: float
    implementation_shortfall_bps: float


@dataclass(frozen=True)
class BacktestFillArtifact:
    """Persisted fill-level artifact row for backtest diagnostics."""

    cycle_ts: datetime
    order_id: uuid.UUID
    instrument_id: uuid.UUID
    side: str
    quantity: int
    requested_quantity: int
    filled_quantity: int
    fill_ratio: float
    raw_fill_price: Decimal
    adjusted_fill_price: Decimal
    commission: Decimal
    adv_shares_20d: float
    participation_pct: float
    spread_bps: float
    slippage_bps: float
    slippage_cost: Decimal
    implementation_shortfall_bps: float
    is_complete: bool


def _compute_annualised_sharpe(
    capital_snapshots: list[Decimal],
    periods_per_year: float = 252.0,
) -> Decimal | None:
    """Annualised Sharpe ratio over a list of per-period NAV snapshots."""
    if len(capital_snapshots) < 2:
        return None

    returns: list[float] = []
    for prev_nav, curr_nav in zip(capital_snapshots[:-1], capital_snapshots[1:], strict=False):
        if prev_nav == 0:
            continue
        returns.append(float((curr_nav - prev_nav) / prev_nav))
    if len(returns) < 20:
        return None

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std <= 0:
        return None

    sharpe = (mean / std) * math.sqrt(periods_per_year)
    return Decimal(str(sharpe)).quantize(Decimal("0.0001"))
