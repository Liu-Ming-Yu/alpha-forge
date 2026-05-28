"""Clone-style vectorized intraday comparator artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.intraday.backtesting.types import (
    IntradayBacktestResult,
)
from quant_platform.services.research_service.intraday.evidence.evidence import (
    _write_intraday_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path


def clone_event_result_as_vectorized(
    event_result: IntradayBacktestResult,
    *,
    output_root: Path,
) -> IntradayBacktestResult:
    """Produce a deterministic vectorized comparator from canonical targets."""
    artifact_root = output_root / f"{event_result.strategy_run_id}_vectorized"
    artifact_root.mkdir(parents=True, exist_ok=True)
    nav_curve = tuple(event_result.nav_curve)
    paths = _write_intraday_artifacts(
        artifact_root,
        spec=None,
        strategy_run_id=event_result.strategy_run_id,
        nav_curve=list(nav_curve),
        fills=[],
        target_weights=dict(event_result.target_weights),
        eligible_universe=dict(event_result.eligible_universe),
        final_capital=event_result.final_capital,
        total_return=event_result.total_return,
        max_drawdown=event_result.max_drawdown,
        engine_name="vectorized_intraday_clone",
    )
    return IntradayBacktestResult(
        strategy_run_id=event_result.strategy_run_id,
        final_capital=event_result.final_capital,
        total_return=event_result.total_return,
        max_drawdown=event_result.max_drawdown,
        nav_curve=nav_curve,
        target_weights=event_result.target_weights,
        eligible_universe=event_result.eligible_universe,
        fills=(),
        residual_order_count=0,
        artifact_root=artifact_root,
        run_summary_uri=paths["run_summary"].resolve().as_uri(),
        execution_quality_uri=paths["execution_quality"].resolve().as_uri(),
        fills_uri=paths["fills"].resolve().as_uri(),
        target_weights_uri=paths["target_weights"].resolve().as_uri(),
    )


__all__ = ["clone_event_result_as_vectorized"]
