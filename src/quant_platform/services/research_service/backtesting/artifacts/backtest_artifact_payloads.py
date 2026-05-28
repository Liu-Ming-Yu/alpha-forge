"""Pure payload builders for backtest artifacts."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from .backtest_artifacts import (
        BacktestCycleMetrics,
        BacktestFillArtifact,
    )


def backtest_parquet_rows(
    *,
    strategy_run_id: uuid.UUID,
    cycle_metrics: list[BacktestCycleMetrics],
    fill_artifacts: list[BacktestFillArtifact],
    empty_timestamp: datetime,
) -> list[dict[str, object]]:
    rows = [_fill_row(strategy_run_id, fill) for fill in fill_artifacts]
    rows.extend(_cycle_row(strategy_run_id, cycle) for cycle in cycle_metrics)
    if not rows:
        rows.append(empty_summary_row(strategy_run_id, empty_timestamp))
    return rows


def execution_quality_payload(
    *,
    fill_artifacts: list[BacktestFillArtifact],
    cycle_metrics: list[BacktestCycleMetrics],
) -> dict[str, object]:
    requested_quantity = sum(f.requested_quantity for f in fill_artifacts)
    filled_quantity = sum(f.filled_quantity for f in fill_artifacts)
    fill_rate = filled_quantity / requested_quantity if requested_quantity > 0 else 0.0
    average_participation = (
        sum(f.participation_pct for f in fill_artifacts) / len(fill_artifacts)
        if fill_artifacts
        else 0.0
    )
    max_participation = max(f.participation_pct for f in fill_artifacts) if fill_artifacts else 0.0
    total_commission = sum((f.commission for f in fill_artifacts), Decimal("0"))
    total_slippage_cost = sum((f.slippage_cost for f in fill_artifacts), Decimal("0"))
    average_shortfall = (
        sum(f.implementation_shortfall_bps for f in fill_artifacts) / len(fill_artifacts)
        if fill_artifacts
        else 0.0
    )

    return {
        "aggregate": {
            "orders_count": sum(c.orders_count for c in cycle_metrics),
            "fills_count": len(fill_artifacts),
            "requested_quantity": requested_quantity,
            "filled_quantity": filled_quantity,
            "fill_rate": fill_rate,
            "average_participation_pct": average_participation,
            "max_participation_pct": max_participation,
            "total_commission": str(total_commission),
            "total_slippage_cost": str(total_slippage_cost),
            "average_implementation_shortfall_bps": average_shortfall,
        },
        "orders": [
            {
                "cycle_ts": f.cycle_ts.isoformat(),
                "order_id": str(f.order_id),
                "instrument_id": str(f.instrument_id),
                "side": f.side,
                "requested_quantity": f.requested_quantity,
                "filled_quantity": f.filled_quantity,
                "fill_ratio": f.fill_ratio,
                "adv_shares_20d": f.adv_shares_20d,
                "participation_pct": f.participation_pct,
                "spread_bps": f.spread_bps,
                "raw_fill_price": str(f.raw_fill_price),
                "adjusted_fill_price": str(f.adjusted_fill_price),
                "commission": str(f.commission),
                "slippage_bps": f.slippage_bps,
                "slippage_cost": str(f.slippage_cost),
                "implementation_shortfall_bps": f.implementation_shortfall_bps,
                "is_complete": f.is_complete,
            }
            for f in fill_artifacts
        ],
    }


def run_summary_payload(
    *,
    initial_capital: Decimal,
    final_capital: Decimal,
    total_return: Decimal,
    annualised_sharpe: Decimal | None,
    max_drawdown: Decimal,
    gross_turnover: Decimal,
    nav_curve: list[Decimal],
) -> dict[str, object]:
    return {
        "initial_capital": str(initial_capital),
        "final_capital": str(final_capital),
        "total_return": str(total_return),
        "annualised_sharpe": None if annualised_sharpe is None else str(annualised_sharpe),
        "max_drawdown": str(max_drawdown),
        "gross_turnover": str(gross_turnover),
        "equity_curve": [float(x) for x in nav_curve],
    }


def empty_summary_row(
    strategy_run_id: uuid.UUID,
    empty_timestamp: datetime,
) -> dict[str, object]:
    return {
        "row_type": "summary",
        "strategy_run_id": str(strategy_run_id),
        "cycle_ts": empty_timestamp,
        "order_id": "",
        "instrument_id": "",
        "side": "",
        "quantity": 0,
        "requested_quantity": 0,
        "filled_quantity": 0,
        "fill_ratio": None,
        "raw_fill_price": None,
        "adjusted_fill_price": None,
        "commission": None,
        "adv_shares_20d": None,
        "participation_pct": None,
        "spread_bps": None,
        "slippage_bps": None,
        "slippage_cost": None,
        "implementation_shortfall_bps": None,
        "is_complete": None,
        "nav": None,
        "signals_count": None,
        "fills_count": None,
        "orders_count": None,
        "fill_rate": None,
    }


def _fill_row(strategy_run_id: uuid.UUID, fill: BacktestFillArtifact) -> dict[str, object]:
    return {
        "row_type": "fill",
        "strategy_run_id": str(strategy_run_id),
        "cycle_ts": fill.cycle_ts,
        "order_id": str(fill.order_id),
        "instrument_id": str(fill.instrument_id),
        "side": fill.side,
        "quantity": fill.quantity,
        "requested_quantity": fill.requested_quantity,
        "filled_quantity": fill.filled_quantity,
        "fill_ratio": fill.fill_ratio,
        "raw_fill_price": float(fill.raw_fill_price),
        "adjusted_fill_price": float(fill.adjusted_fill_price),
        "commission": float(fill.commission),
        "adv_shares_20d": fill.adv_shares_20d,
        "participation_pct": fill.participation_pct,
        "spread_bps": fill.spread_bps,
        "slippage_bps": fill.slippage_bps,
        "slippage_cost": float(fill.slippage_cost),
        "implementation_shortfall_bps": fill.implementation_shortfall_bps,
        "is_complete": fill.is_complete,
        "nav": None,
        "signals_count": None,
        "fills_count": None,
        "orders_count": None,
        "fill_rate": None,
    }


def _cycle_row(strategy_run_id: uuid.UUID, cycle: BacktestCycleMetrics) -> dict[str, object]:
    return {
        "row_type": "cycle",
        "strategy_run_id": str(strategy_run_id),
        "cycle_ts": cycle.timestamp,
        "order_id": "",
        "instrument_id": "",
        "side": "",
        "quantity": 0,
        "requested_quantity": 0,
        "filled_quantity": 0,
        "fill_ratio": None,
        "raw_fill_price": None,
        "adjusted_fill_price": None,
        "commission": float(cycle.total_commission),
        "adv_shares_20d": None,
        "participation_pct": cycle.average_participation_pct,
        "spread_bps": None,
        "slippage_bps": cycle.total_slippage_bps,
        "slippage_cost": None,
        "implementation_shortfall_bps": cycle.implementation_shortfall_bps,
        "is_complete": None,
        "nav": float(cycle.nav),
        "signals_count": cycle.signals_count,
        "fills_count": cycle.fills_count,
        "orders_count": cycle.orders_count,
        "fill_rate": cycle.fill_rate,
    }


__all__ = [
    "backtest_parquet_rows",
    "empty_summary_row",
    "execution_quality_payload",
    "run_summary_payload",
]
