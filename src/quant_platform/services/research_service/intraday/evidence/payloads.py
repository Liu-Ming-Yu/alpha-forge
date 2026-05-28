"""Pure payload builders for intraday backtest evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime

    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.core.domain.research import (
        BacktestEvidenceManifest,
        BacktestReconciliationReport,
        IntradayBacktestSpec,
    )
    from quant_platform.services.research_service.intraday.backtesting.types import (
        IntradayFillArtifact,
    )


def fill_payload(fill: IntradayFillArtifact) -> dict[str, object]:
    return {
        "order_id": str(fill.order_id),
        "instrument_id": str(fill.instrument_id),
        "side": fill.side,
        "tactic": fill.tactic,
        "executed_at": fill.executed_at.isoformat(),
        "quantity": fill.quantity,
        "requested_quantity": fill.requested_quantity,
        "residual_quantity": fill.residual_quantity,
        "arrival_price": str(fill.arrival_price),
        "decision_price": str(fill.decision_price),
        "minute_vwap": str(fill.minute_vwap),
        "fill_price": str(fill.fill_price),
        "spread_bps": str(fill.spread_bps),
        "participation_rate": str(fill.participation_rate),
        "slippage_bps": str(fill.slippage_bps),
        "implementation_shortfall_bps": str(fill.implementation_shortfall_bps),
        "commission": str(fill.commission),
        "is_complete": fill.is_complete,
    }


def spec_payload(spec: IntradayBacktestSpec) -> dict[str, object]:
    return {
        "strategy_name": spec.strategy_name,
        "strategy_version": spec.strategy_version,
        "start": spec.start.isoformat(),
        "end": spec.end.isoformat(),
        "initial_capital": str(spec.initial_capital),
        "decision_times": [ts.isoformat() for ts in spec.decision_times],
        "dataset_ids": [str(item) for item in spec.dataset_ids],
        "universe_name": spec.universe_name,
        "feature_set_version": spec.feature_set_version,
        "model_version": spec.model_version,
        "execution_profile": spec.execution_profile,
        "benchmark_instrument_id": str(spec.benchmark_instrument_id)
        if spec.benchmark_instrument_id
        else None,
    }


def manifest_payload(manifest: BacktestEvidenceManifest) -> dict[str, object]:
    return {
        "manifest_id": str(manifest.manifest_id),
        "strategy_run_id": str(manifest.strategy_run_id),
        "created_at": manifest.created_at.isoformat(),
        "spec": spec_payload(manifest.spec),
        "code_commit": manifest.code_commit,
        "config_hash": manifest.config_hash,
        "dataset_ids": [str(item) for item in manifest.dataset_ids],
        "universe_snapshot_id": str(manifest.universe_snapshot_id),
        "feature_dataset_id": str(manifest.feature_dataset_id)
        if manifest.feature_dataset_id
        else None,
        "model_artifact_id": str(manifest.model_artifact_id)
        if manifest.model_artifact_id
        else None,
        "calibration_artifact_uri": manifest.calibration_artifact_uri,
        "execution_quality_uri": manifest.execution_quality_uri,
        "reconciliation_report_uri": manifest.reconciliation_report_uri,
        "event_driven_artifact_uri": manifest.event_driven_artifact_uri,
        "vectorized_artifact_uri": manifest.vectorized_artifact_uri,
        "passed": manifest.passed,
        "blockers": list(manifest.blockers),
    }


def reconciliation_payload(report: BacktestReconciliationReport) -> dict[str, object]:
    return {
        "report_id": str(report.report_id),
        "strategy_run_id": str(report.strategy_run_id),
        "generated_at": report.generated_at.isoformat(),
        "status": report.status.value,
        "passed": report.passed,
        "comparable": report.comparable,
        "target_weight_max_diff_bps": str(report.target_weight_max_diff_bps),
        "nav_diff_bps": str(report.nav_diff_bps),
        "max_drawdown_diff_bps": str(report.max_drawdown_diff_bps),
        "tolerance_target_weight_bps": str(report.tolerance_target_weight_bps),
        "tolerance_nav_bps": str(report.tolerance_nav_bps),
        "tolerance_drawdown_bps": str(report.tolerance_drawdown_bps),
        "residual_order_count": report.residual_order_count,
        "missing_artifacts": list(report.missing_artifacts),
        "breaches": list(report.breaches),
    }


def stable_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def target_weights_payload(
    *,
    target_weights: Mapping[datetime, Mapping[uuid.UUID, Decimal]],
    eligible_universe: Mapping[datetime, tuple[uuid.UUID, ...]],
) -> dict[str, object]:
    return {
        "target_weights": {
            ts.isoformat(): {str(iid): str(weight) for iid, weight in weights.items()}
            for ts, weights in target_weights.items()
        },
        "eligible_universe": {
            ts.isoformat(): [str(iid) for iid in ids] for ts, ids in eligible_universe.items()
        },
    }


def intraday_execution_quality_payload(
    fills: list[IntradayFillArtifact],
) -> dict[str, object]:
    fills_payload = [fill_payload(fill) for fill in fills]
    total_requested = sum(fill.requested_quantity for fill in fills)
    total_filled = sum(fill.quantity for fill in fills)
    return {
        "aggregate": {
            "fills_count": len(fills),
            "requested_quantity": total_requested,
            "filled_quantity": total_filled,
            "fill_rate": total_filled / total_requested if total_requested else 0.0,
            "average_participation_pct": _mean_float(
                [float(fill.participation_rate) for fill in fills]
            ),
            "average_implementation_shortfall_bps": _mean_float(
                [float(fill.implementation_shortfall_bps) for fill in fills]
            ),
            "total_commission": str(sum((fill.commission for fill in fills), Decimal("0"))),
            "residual_fill_rows": sum(1 for fill in fills if not fill.is_complete),
        },
        "orders": fills_payload,
    }


def intraday_run_summary_payload(
    *,
    spec: IntradayBacktestSpec | None,
    strategy_run_id: uuid.UUID,
    nav_curve: list[tuple[datetime, Decimal]],
    final_capital: Decimal,
    total_return: Decimal,
    max_drawdown: Decimal,
    engine_name: str,
    engine_version: str,
    input_hash: str,
    cost_assumptions: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "strategy_run_id": str(strategy_run_id),
        "engine_name": engine_name,
        "engine_version": engine_version,
        "input_hash": input_hash,
        "cost_assumptions": dict(cost_assumptions or {}),
        "spec": spec_payload(spec) if spec is not None else None,
        "final_capital": str(final_capital),
        "total_return": str(total_return),
        "max_drawdown": str(max_drawdown),
        "equity_curve": [str(nav) for _, nav in nav_curve],
        "nav_curve": [{"timestamp": ts.isoformat(), "nav": str(nav)} for ts, nav in nav_curve],
    }


def intraday_input_hash(
    spec: IntradayBacktestSpec,
    feature_series: Mapping[datetime, Mapping[uuid.UUID, Mapping[str, float]]],
    minute_bars: Mapping[uuid.UUID, list[MarketBar]],
) -> str:
    payload = {
        "spec": spec_payload(spec),
        "features": {
            ts.isoformat(): {
                str(instrument_id): {str(name): float(value) for name, value in values.items()}
                for instrument_id, values in instruments.items()
            }
            for ts, instruments in feature_series.items()
        },
        "bars": {
            str(instrument_id): [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": str(bar.open),
                    "high": str(bar.high),
                    "low": str(bar.low),
                    "close": str(bar.close),
                    "volume": bar.volume,
                    "vwap": str(bar.vwap) if bar.vwap is not None else None,
                }
                for bar in bars
            ]
            for instrument_id, bars in minute_bars.items()
        },
    }
    return stable_hash(payload)


def _mean_float(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
