"""Backtest artifact serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from .backtest_artifact_payloads import (
    backtest_parquet_rows,
    execution_quality_payload,
    run_summary_payload,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from .backtest_artifacts import (
        BacktestCycleMetrics,
        BacktestFillArtifact,
    )


def write_backtest_artifacts(
    object_store_root: str | Path,
    strategy_run_id: uuid.UUID,
    cycle_metrics: list[BacktestCycleMetrics],
    fill_artifacts: list[BacktestFillArtifact],
    *,
    empty_timestamp: datetime,
) -> str:
    """Persist cycle and fill diagnostics to the shared Parquet artifact schema."""
    artifact_dir = Path(object_store_root) / "backtests"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{strategy_run_id}.parquet"

    rows = backtest_parquet_rows(
        strategy_run_id=strategy_run_id,
        cycle_metrics=cycle_metrics,
        fill_artifacts=fill_artifacts,
        empty_timestamp=empty_timestamp,
    )
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, artifact_path)
    return artifact_path.resolve().as_uri()


def write_execution_quality(
    object_store_root: str | Path,
    strategy_run_id: uuid.UUID,
    *,
    fill_artifacts: list[BacktestFillArtifact],
    cycle_metrics: list[BacktestCycleMetrics],
) -> Path:
    """Write execution-quality JSON consumed by the tearsheet."""
    run_dir = _tearsheet_run_dir(object_store_root, strategy_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "execution_quality.json"

    payload = execution_quality_payload(
        fill_artifacts=fill_artifacts,
        cycle_metrics=cycle_metrics,
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_run_summary(
    object_store_root: str | Path,
    strategy_run_id: uuid.UUID,
    *,
    initial_capital: Decimal,
    final_capital: Decimal,
    total_return: Decimal,
    annualised_sharpe: Decimal | None,
    max_drawdown: Decimal,
    gross_turnover: Decimal,
    nav_curve: list[Decimal],
) -> Path:
    """Write the risk-summary JSON sidecar consumed by the tearsheet."""
    run_dir = _tearsheet_run_dir(object_store_root, strategy_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_summary.json"
    payload = run_summary_payload(
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        annualised_sharpe=annualised_sharpe,
        max_drawdown=max_drawdown,
        gross_turnover=gross_turnover,
        nav_curve=nav_curve,
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _tearsheet_run_dir(object_store_root: str | Path, strategy_run_id: uuid.UUID) -> Path:
    return Path(object_store_root) / "tearsheets" / str(strategy_run_id)
