"""Pure status calculations for the performance repository package."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import (
    NavSnapshot,
    PerformanceReport,
    ShadowPaperParityRecord,
    ShadowPaperParityStatus,
    SignalGateRecord,
    SignalGateStatus,
    TextSignalGateRecord,
    TextSignalGateStatus,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


def build_performance_report(
    strategy_run_id: uuid.UUID,
    *,
    as_of: datetime,
    rows: list[NavSnapshot],
) -> PerformanceReport:
    navs = [row.net_asset_value for row in rows if row.net_asset_value > 0]
    returns: list[float] = []
    for prev, curr in zip(navs[:-1], navs[1:], strict=False):
        if prev > 0:
            returns.append(float((curr - prev) / prev))
    rolling_sharpe = 0.0
    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        if std > 0:
            rolling_sharpe = (mean / std) * math.sqrt(252)

    max_drawdown = 0.0
    if navs:
        peak = navs[0]
        for nav in navs[1:]:
            if nav > peak:
                peak = nav
            if peak > 0:
                drawdown = float((nav - peak) / peak)
                max_drawdown = min(max_drawdown, drawdown)

    gross_turnover = 0.0
    if rows:
        exposures = [float(row.gross_exposure) for row in rows]
        gross_turnover = sum(
            abs(curr - prev) for prev, curr in zip(exposures[:-1], exposures[1:], strict=False)
        )

    return PerformanceReport(
        strategy_run_id=strategy_run_id,
        as_of=as_of,
        observations=len(rows),
        rolling_sharpe=rolling_sharpe,
        max_drawdown=max_drawdown,
        gross_turnover=gross_turnover,
        rolling_ic=0.0,
        slippage_ratio=1.0,
    )


def build_text_gate_status(
    strategy_name: str,
    *,
    as_of: datetime,
    records: list[TextSignalGateRecord],
    min_observations: int,
    min_ic: float,
    max_negative_streak: int,
) -> TextSignalGateStatus:
    recent = records[-min_observations:] if min_observations > 0 else records
    weighted_count = sum(max(1, row.observations) for row in recent)
    rolling_ic = 0.0
    if weighted_count > 0:
        rolling_ic = sum(row.daily_ic * max(1, row.observations) for row in recent) / weighted_count
    negative_streak = 0
    for row in reversed(records):
        if row.daily_ic < 0:
            negative_streak += 1
        else:
            break
    return TextSignalGateStatus(
        strategy_name=strategy_name,
        as_of=as_of,
        observations=len(records),
        rolling_ic=rolling_ic,
        negative_streak=negative_streak,
        min_observations=min_observations,
        min_ic=min_ic,
        max_negative_streak=max_negative_streak,
    )


def build_signal_gate_status(
    signal_name: str,
    signal_type: str,
    *,
    as_of: datetime,
    records: list[SignalGateRecord],
    min_observations: int,
    min_ic: float,
    max_negative_streak: int,
    drawdown_limit: float,
    turnover_limit: float,
) -> SignalGateStatus:
    recent = records[-min_observations:] if min_observations > 0 else records
    weighted_count = sum(max(1, row.observations) for row in recent)
    rolling_ic = 0.0
    if weighted_count > 0:
        rolling_ic = sum(row.daily_ic * max(1, row.observations) for row in recent) / weighted_count
    negative_streak = 0
    for row in reversed(records):
        if row.daily_ic < 0:
            negative_streak += 1
        else:
            break
    max_drawdown = min((row.drawdown for row in records), default=0.0)
    max_turnover = max((row.turnover for row in records), default=0.0)
    return SignalGateStatus(
        signal_name=signal_name,
        signal_type=signal_type,
        as_of=as_of,
        observations=len(records),
        rolling_ic=rolling_ic,
        negative_streak=negative_streak,
        max_drawdown=max_drawdown,
        max_turnover=max_turnover,
        min_observations=min_observations,
        min_ic=min_ic,
        max_negative_streak=max_negative_streak,
        drawdown_limit=drawdown_limit,
        turnover_limit=turnover_limit,
    )


def build_shadow_paper_parity_status(
    signal_name: str,
    signal_type: str,
    *,
    as_of: datetime,
    records: list[ShadowPaperParityRecord],
    min_trading_days: int,
    max_target_weight_diff_bps: float,
) -> ShadowPaperParityStatus:
    """Aggregate strict shadow-vs-paper parity evidence."""
    filtered = [row for row in records if row.as_of <= as_of]
    trading_days = {row.trading_day for row in filtered}
    return ShadowPaperParityStatus(
        signal_name=signal_name,
        signal_type=signal_type,
        as_of=as_of,
        observations=len(filtered),
        trading_days=len(trading_days),
        min_trading_days=min_trading_days,
        max_target_weight_diff_bps=max(
            (row.max_target_weight_diff_bps for row in filtered),
            default=0.0,
        ),
        max_allowed_target_weight_diff_bps=max_target_weight_diff_bps,
        missing_instruments=sum(row.missing_instruments for row in filtered),
        order_side_mismatches=sum(row.order_side_mismatches for row in filtered),
    )


_build_performance_report = build_performance_report
_build_signal_gate_status = build_signal_gate_status
_build_shadow_paper_parity_status = build_shadow_paper_parity_status
_build_text_gate_status = build_text_gate_status
