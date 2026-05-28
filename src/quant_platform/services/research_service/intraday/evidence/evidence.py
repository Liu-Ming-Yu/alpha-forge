"""Intraday backtest evidence and artifact serialization."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import (
    BacktestReconciliationReport,
    BacktestReconciliationStatus,
)
from quant_platform.services.research_service.intraday.evidence.artifacts import (
    assert_backtest_evidence,
    write_backtest_evidence_manifest,
    write_reconciliation_report,
)
from quant_platform.services.research_service.intraday.evidence.artifacts import (
    write_intraday_artifacts as _write_intraday_artifacts,
)
from quant_platform.services.research_service.intraday.evidence.payloads import (
    intraday_input_hash as _intraday_input_hash,
)

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.services.research_service.intraday.backtesting.types import (
        IntradayBacktestResult,
    )

__all__ = [
    "_intraday_input_hash",
    "_write_intraday_artifacts",
    "assert_backtest_evidence",
    "reconcile_intraday_backtests",
    "write_backtest_evidence_manifest",
    "write_reconciliation_report",
]


def reconcile_intraday_backtests(
    *,
    event_result: IntradayBacktestResult,
    vectorized_result: IntradayBacktestResult,
    generated_at: datetime,
    tolerance_target_weight_bps: Decimal = Decimal("1"),
    tolerance_nav_bps: Decimal = Decimal("50"),
    tolerance_drawdown_bps: Decimal = Decimal("50"),
) -> BacktestReconciliationReport:
    """Compare canonical event replay with vectorized replay."""
    breaches: list[str] = []
    missing: list[str] = []
    for label, result in (("event", event_result), ("vectorized", vectorized_result)):
        for name, uri in (
            ("run_summary", result.run_summary_uri),
            ("target_weights", result.target_weights_uri),
            ("execution_quality", result.execution_quality_uri),
        ):
            if not uri:
                missing.append(f"{label}:{name}")
    universe_diff = _eligible_universe_diff(event_result, vectorized_result)
    if universe_diff:
        breaches.append(universe_diff)
    target_diff = _target_weight_max_diff_bps(event_result, vectorized_result)
    if target_diff > tolerance_target_weight_bps:
        breaches.append(f"target_weight_diff_bps={target_diff}")
    nav_diff = _nav_diff_bps(event_result, vectorized_result)
    if nav_diff > tolerance_nav_bps:
        breaches.append(f"nav_diff_bps={nav_diff}")
    dd_diff = abs(event_result.max_drawdown - vectorized_result.max_drawdown) * Decimal("10000")
    if dd_diff > tolerance_drawdown_bps:
        breaches.append(f"max_drawdown_diff_bps={dd_diff}")
    comparable = event_result.residual_order_count == 0 and not missing
    if event_result.residual_order_count:
        breaches.append(f"residual_orders={event_result.residual_order_count}")
    if missing:
        breaches.append(f"missing_artifacts={','.join(missing)}")
    if not comparable:
        status = BacktestReconciliationStatus.NON_COMPARABLE
    elif breaches:
        status = BacktestReconciliationStatus.FAILED
    else:
        status = BacktestReconciliationStatus.PASSED
    return BacktestReconciliationReport(
        report_id=uuid.uuid4(),
        strategy_run_id=event_result.strategy_run_id,
        generated_at=generated_at,
        status=status,
        passed=status == BacktestReconciliationStatus.PASSED,
        comparable=comparable,
        target_weight_max_diff_bps=target_diff,
        nav_diff_bps=nav_diff,
        max_drawdown_diff_bps=dd_diff,
        tolerance_target_weight_bps=tolerance_target_weight_bps,
        tolerance_nav_bps=tolerance_nav_bps,
        tolerance_drawdown_bps=tolerance_drawdown_bps,
        residual_order_count=event_result.residual_order_count,
        missing_artifacts=tuple(missing),
        breaches=tuple(breaches),
    )


def _target_weight_max_diff_bps(
    left: IntradayBacktestResult,
    right: IntradayBacktestResult,
) -> Decimal:
    worst = Decimal("0")
    timestamps = set(left.target_weights) | set(right.target_weights)
    for ts in timestamps:
        lw = left.target_weights.get(ts, {})
        rw = right.target_weights.get(ts, {})
        for iid in set(lw) | set(rw):
            worst = max(
                worst, abs(lw.get(iid, Decimal("0")) - rw.get(iid, Decimal("0"))) * Decimal("10000")
            )
    return worst


def _nav_diff_bps(left: IntradayBacktestResult, right: IntradayBacktestResult) -> Decimal:
    if not left.nav_curve or left.nav_curve[0][1] <= 0:
        return Decimal("0")
    return abs(left.final_capital - right.final_capital) / left.nav_curve[0][1] * Decimal("10000")


def _eligible_universe_diff(
    left: IntradayBacktestResult,
    right: IntradayBacktestResult,
) -> str:
    for ts in set(left.eligible_universe) | set(right.eligible_universe):
        if set(left.eligible_universe.get(ts, ())) != set(right.eligible_universe.get(ts, ())):
            return f"eligible_universe_diff={ts.isoformat()}"
    return ""
