"""Row-to-domain mappers for the performance repository package."""

from __future__ import annotations

import json
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from quant_platform.core.domain.production import (
    BrokerHealthObservation,
    BrokerSmokeObservation,
    InstrumentPnl,
    MetricRollupSnapshot,
    NavSnapshot,
    PaperLifecycleObservation,
    PredictionResult,
    RuntimeHeartbeat,
    ShadowPaperParityRecord,
    SignalGateRecord,
    TextSignalGateRecord,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime


def _metadata(raw: object) -> dict[str, object]:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


def _optional_decimal(value: object) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def row_to_nav(row: object) -> NavSnapshot:
    row = cast("Mapping[str, Any]", row)
    realized = row.get("realized_pnl")
    unrealized = row.get("unrealized_pnl")
    return NavSnapshot(
        snapshot_id=uuid.UUID(str(row["snapshot_id"])),
        strategy_run_id=uuid.UUID(str(row["strategy_run_id"])),
        as_of=cast("datetime", row["as_of"]),
        net_asset_value=Decimal(str(row["net_asset_value"])),
        gross_exposure=Decimal(str(row["gross_exposure"])),
        cash=Decimal(str(row["cash"])),
        source=str(row["source"]),
        realized_pnl=_optional_decimal(realized),
        unrealized_pnl=_optional_decimal(unrealized),
    )


def row_to_instrument_pnl(row: object) -> InstrumentPnl:
    row = cast("Mapping[str, Any]", row)
    return InstrumentPnl(
        pnl_id=uuid.UUID(str(row["pnl_id"])),
        strategy_run_id=uuid.UUID(str(row["strategy_run_id"])),
        instrument_id=uuid.UUID(str(row["instrument_id"])),
        as_of=cast("datetime", row["as_of"]),
        realized_pnl=Decimal(str(row["realized_pnl"])),
        unrealized_pnl=Decimal(str(row["unrealized_pnl"])),
        weight=Decimal(str(row["weight"])),
        contribution=Decimal(str(row["contribution"])),
    )


def row_to_metric_rollup(row: object) -> MetricRollupSnapshot:
    row = cast("Mapping[str, Any]", row)
    labels = _metadata(row["labels_json"])
    return MetricRollupSnapshot(
        snapshot_id=uuid.UUID(str(row["snapshot_id"])),
        metric_name=str(row["metric_name"]),
        as_of=cast("datetime", row["as_of"]),
        window=str(row["window"]),
        value=float(row["value"]),
        labels={str(key): str(value) for key, value in labels.items()},
        source=str(row["source"]),
    )


def row_to_ic(row: object) -> TextSignalGateRecord:
    row = cast("Mapping[str, Any]", row)
    return TextSignalGateRecord(
        strategy_name=str(row["strategy_name"]),
        as_of=cast("datetime", row["as_of"]),
        daily_ic=float(row["daily_ic"]),
        observations=int(row["observations"]),
        metadata=_metadata(row["metadata_json"]),
    )


def row_to_signal(row: object) -> SignalGateRecord:
    row = cast("Mapping[str, Any]", row)
    return SignalGateRecord(
        signal_name=str(row["signal_name"]),
        signal_type=str(row["signal_type"]),
        as_of=cast("datetime", row["as_of"]),
        daily_ic=float(row["daily_ic"]),
        observations=int(row["observations"]),
        drawdown=float(row["drawdown"]),
        turnover=float(row["turnover"]),
        metadata=_metadata(row["metadata_json"]),
    )


def row_to_prediction(row: object) -> PredictionResult:
    row = cast("Mapping[str, Any]", row)
    blockers = row.get("blockers_json", ())
    if isinstance(blockers, str):
        blockers = json.loads(blockers)
    if not isinstance(blockers, list | tuple):
        blockers = ()
    return PredictionResult(
        prediction_id=uuid.UUID(str(row["prediction_id"])),
        strategy_run_id=uuid.UUID(str(row["strategy_run_id"])),
        instrument_id=uuid.UUID(str(row["instrument_id"])),
        source=str(row["source"]),
        model_version=str(row["model_version"]),
        as_of=cast("datetime", row["as_of"]),
        horizon=str(row["horizon"]),
        expected_return=float(row["expected_return"]),
        rank_score=float(row["rank_score"]),
        confidence=float(row["confidence"]),
        feature_schema_hash=str(row["feature_schema_hash"]),
        calibration_bucket=str(row["calibration_bucket"]),
        blockers=tuple(str(item) for item in blockers),
        metadata=_metadata(row["metadata_json"]),
    )


def row_to_shadow_paper_parity(row: object) -> ShadowPaperParityRecord:
    row = cast("Mapping[str, Any]", row)
    metadata = _metadata(row["metadata_json"])
    return ShadowPaperParityRecord(
        parity_id=uuid.UUID(str(row["parity_id"])),
        signal_name=str(row["signal_name"]),
        signal_type=str(row["signal_type"]),
        trading_day=row["trading_day"],
        as_of=cast("datetime", row["as_of"]),
        instruments_compared=int(row["instruments_compared"]),
        missing_instruments=int(row["missing_instruments"]),
        max_target_weight_diff_bps=float(row["max_target_weight_diff_bps"]),
        order_side_mismatches=int(row["order_side_mismatches"]),
        metadata=metadata,
    )


def row_to_heartbeat(row: object) -> RuntimeHeartbeat:
    row = cast("Mapping[str, Any]", row)
    return RuntimeHeartbeat(
        component=str(row["component"]),
        as_of=cast("datetime", row["as_of"]),
        status=str(row["status"]),
        detail=str(row["detail"] or ""),
    )


def row_to_broker_health(row: object) -> BrokerHealthObservation:
    row = cast("Mapping[str, Any]", row)
    return BrokerHealthObservation(
        observed_at=cast("datetime", row["observed_at"]),
        status=str(row["status"]),
        latency_ms=float(row["latency_ms"]),
        last_heartbeat_at=cast("datetime | None", row["last_heartbeat_at"]),
        detail=str(row["detail"] or ""),
    )


def row_to_broker_smoke(row: object) -> BrokerSmokeObservation:
    row = cast("Mapping[str, Any]", row)
    return BrokerSmokeObservation(
        observed_at=cast("datetime", row["observed_at"]),
        status=str(row["status"]),
        host=str(row["host"]),
        port=int(row["port"]),
        client_id=int(row["client_id"]),
        latency_ms=float(row["latency_ms"]),
        account_status=str(row["account_status"]),
        positions_status=str(row["positions_status"]),
        open_orders_status=str(row["open_orders_status"]),
        detail=str(row["detail"] or ""),
    )


def row_to_paper_lifecycle(row: object) -> PaperLifecycleObservation:
    row = cast("Mapping[str, Any]", row)
    return PaperLifecycleObservation(
        observed_at=cast("datetime", row["observed_at"]),
        status=str(row["status"]),
        host=str(row["host"]),
        port=int(row["port"]),
        client_id=int(row["client_id"]),
        instrument_id=uuid.UUID(str(row["instrument_id"])),
        broker_order_id=str(row["broker_order_id"]),
        max_notional_usd=Decimal(str(row["max_notional_usd"])),
        limit_price=Decimal(str(row["limit_price"])),
        quantity=int(row["quantity"]),
        ack_status=str(row["ack_status"]),
        cancel_status=str(row["cancel_status"]),
        stale_open_order_count=int(row["stale_open_order_count"]),
        detail=str(row["detail"] or ""),
    )


_row_to_broker_health = row_to_broker_health
_row_to_broker_smoke = row_to_broker_smoke
_row_to_heartbeat = row_to_heartbeat
_row_to_ic = row_to_ic
_row_to_instrument_pnl = row_to_instrument_pnl
_row_to_metric_rollup = row_to_metric_rollup
_row_to_nav = row_to_nav
_row_to_paper_lifecycle = row_to_paper_lifecycle
_row_to_prediction = row_to_prediction
_row_to_shadow_paper_parity = row_to_shadow_paper_parity
_row_to_signal = row_to_signal
