"""Prometheus metrics facade and typed helper functions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import structlog

from quant_platform.infrastructure.metrics.catalog import (
    TEXT_SIGNAL_IC_ROLLING_20D,
    alpha_ensemble_live_cap,
    alpha_live_ramp_level,
    alpha_source_errors_total,
    alpha_source_score_coverage,
    alpha_text_extraction_latency_seconds,
    alpha_text_ingestion_lag_seconds,
    bar_gaps_detected_total,
    bar_read_latency_seconds,
    broker_operation_latency_seconds,
    cash_drift_usd,
    cash_reservation_age_seconds,
    cycle_duration_seconds,
    cycle_errors_total,
    db_pool_utilization,
    db_query_latency_seconds,
    event_bus_dead_letter_depth,
    event_bus_dead_letter_total,
    event_bus_pending_entries_total,
    event_bus_publish_total,
    event_bus_stream_length,
    event_deser_errors_total,
    feature_compute_latency_seconds,
    feature_mean,
    feature_nan_total,
    feature_std,
    fill_latency_seconds,
    fill_rate_pct,
    http_request_duration_seconds,
    http_requests_total,
    kill_switch_active,
    lock_lease_lost_total,
    lock_operations_total,
    order_ack_latency_seconds,
    order_rejected_total,
    order_submit_latency_seconds,
    order_submitted_total,
    pg_pool_checked_out,
    realized_slippage_bps,
    reconciliation_mismatches_total,
    redis_connection_errors_total,
    risk_gate_rejections_total,
    throttle_tokens,
)
from quant_platform.infrastructure.metrics.registry import (
    NOOP as _REGISTRY_NOOP,
)
from quant_platform.infrastructure.metrics.registry import (
    MetricLike as _MetricLike,
)
from quant_platform.infrastructure.metrics.registry import (
    get_registry,
)
from quant_platform.infrastructure.metrics.registry import (
    render_latest as _render_latest,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

log = structlog.get_logger(__name__)

_NOOP: _MetricLike = _REGISTRY_NOOP
__all__ = [
    "TEXT_SIGNAL_IC_ROLLING_20D",
    "_NOOP",
    "get_registry",
    "render_latest",
]


def render_latest() -> tuple[bytes, str]:
    """Return the current Prometheus exposition payload."""
    return _render_latest()


@contextmanager
def time_phase(engine: str, phase: str) -> Iterator[None]:
    """Record the wall-time of a strategy-cycle phase."""
    import time

    start = time.monotonic()
    try:
        yield
    finally:
        cycle_duration_seconds.labels(engine=engine, phase=phase).observe(time.monotonic() - start)


@contextmanager
def time_db_operation(operation: str) -> Iterator[None]:
    """Record the wall-time of a PostgreSQL repository call."""
    import time

    start = time.monotonic()
    try:
        yield
    finally:
        db_query_latency_seconds.labels(operation=operation).observe(time.monotonic() - start)


@contextmanager
def time_broker_operation(operation: str) -> Iterator[None]:
    """Record the wall-time of a broker gateway call."""
    import time

    start = time.monotonic()
    try:
        yield
    finally:
        broker_operation_latency_seconds.labels(operation=operation).observe(
            time.monotonic() - start
        )


def record_cycle_error(engine: str) -> None:
    cycle_errors_total.labels(engine=engine).inc()


def set_kill_switch(active: bool) -> None:
    kill_switch_active.set(1.0 if active else 0.0)


def record_lock_operation(resource: str, op: str, outcome: str) -> None:
    lock_operations_total.labels(resource=resource, op=op, outcome=outcome).inc()


def record_lease_loss(resource: str) -> None:
    lock_lease_lost_total.labels(resource=resource).inc()
    log.error("distributed_lock.lease_lost", resource=resource)


def record_event_publish(backend: str, outcome: str) -> None:
    event_bus_publish_total.labels(backend=backend, outcome=outcome).inc()


def set_pending_entries(backend: str, stream: str, count: int) -> None:
    """Report Redis Stream length for publish-side observability."""
    event_bus_stream_length.labels(backend=backend, stream=stream).set(float(count))


def set_stream_length(backend: str, stream: str, count: int) -> None:
    """Report total Redis Stream entry count."""
    event_bus_stream_length.labels(backend=backend, stream=stream).set(float(count))


def set_pending_entries_total(backend: str, stream: str, group: str, count: int) -> None:
    """Report authoritative Redis consumer-group PEL depth."""
    event_bus_pending_entries_total.labels(backend=backend, stream=stream, group=group).set(
        float(count)
    )


def record_dead_letter(backend: str, stream: str, group: str) -> None:
    event_bus_dead_letter_total.labels(backend=backend, stream=stream, group=group).inc()


def set_dead_letter_depth(backend: str, stream: str, depth: int) -> None:
    event_bus_dead_letter_depth.labels(backend=backend, stream=stream).set(max(0, int(depth)))


def record_order_submitted(engine: str) -> None:
    order_submitted_total.labels(engine=engine).inc()


def record_order_rejected(engine: str, reason: str) -> None:
    order_rejected_total.labels(engine=engine, reason=reason).inc()


def observe_fill_latency(engine: str, seconds: float) -> None:
    fill_latency_seconds.labels(engine=engine).observe(max(0.0, seconds))


def observe_order_submit_latency(engine: str, outcome: str, seconds: float) -> None:
    order_submit_latency_seconds.labels(engine=engine, outcome=outcome).observe(max(0.0, seconds))


def set_cash_drift(engine: str, drift_usd: float) -> None:
    cash_drift_usd.labels(engine=engine).set(drift_usd)


def record_reconciliation_mismatch(engine: str, mismatch_type: str) -> None:
    reconciliation_mismatches_total.labels(engine=engine, type=mismatch_type).inc()


def set_throttle_tokens(tokens: float) -> None:
    throttle_tokens.set(tokens)


def set_db_pool_utilization(checked_out: int, idle: int, overflow: int) -> None:
    db_pool_utilization.labels(state="checked_out").set(float(checked_out))
    db_pool_utilization.labels(state="idle").set(float(idle))
    db_pool_utilization.labels(state="overflow").set(float(overflow))


def record_risk_gate_rejection(reason: str) -> None:
    risk_gate_rejections_total.labels(reason=reason).inc()


def set_cash_reservation_age(age_seconds: float) -> None:
    cash_reservation_age_seconds.set(max(0.0, age_seconds))


def record_http_request(method: str, endpoint: str, status: int, duration_seconds: float) -> None:
    http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
        max(0.0, duration_seconds)
    )


def set_alpha_source_coverage(source: str, mode: str, coverage: float) -> None:
    alpha_source_score_coverage.labels(source=source, mode=mode).set(max(0.0, min(1.0, coverage)))


def record_alpha_source_error(source: str, error_type: str) -> None:
    alpha_source_errors_total.labels(source=source, error_type=error_type).inc()


def observe_alpha_text_extraction_latency(model: str, outcome: str, seconds: float) -> None:
    alpha_text_extraction_latency_seconds.labels(model=model, outcome=outcome).observe(
        max(0.0, seconds)
    )


def set_alpha_text_ingestion_lag(provider: str, seconds: float) -> None:
    alpha_text_ingestion_lag_seconds.labels(provider=provider).set(max(0.0, seconds))


def set_alpha_live_controls(*, cap: float, ramp_level: float) -> None:
    alpha_ensemble_live_cap.set(max(0.0, min(1.0, cap)))
    alpha_live_ramp_level.set(max(0.0, min(1.0, ramp_level)))


def observe_order_ack_latency(engine: str, seconds: float) -> None:
    order_ack_latency_seconds.labels(engine=engine).observe(max(0.0, seconds))


def observe_bar_read_latency(operation: str, seconds: float) -> None:
    bar_read_latency_seconds.labels(operation=operation).observe(max(0.0, seconds))


def observe_feature_compute_latency(source: str, seconds: float) -> None:
    feature_compute_latency_seconds.labels(source=source).observe(max(0.0, seconds))


def observe_realized_slippage(engine: str, bps: float) -> None:
    realized_slippage_bps.labels(engine=engine).observe(bps)


def set_fill_rate(engine: str, rate: float) -> None:
    fill_rate_pct.labels(engine=engine).set(max(0.0, min(1.0, rate)))


def record_bar_gap(instrument_id: str) -> None:
    bar_gaps_detected_total.labels(instrument_id=instrument_id).inc()


def record_event_deser_error(event_type: str) -> None:
    event_deser_errors_total.labels(event_type=event_type).inc()


def set_pg_pool_checked_out(count: int) -> None:
    pg_pool_checked_out.set(float(max(0, count)))


def record_redis_connection_error() -> None:
    redis_connection_errors_total.inc()


def set_feature_mean(feature_name: str, feature_set_version: str, value: float) -> None:
    feature_mean.labels(feature_name=feature_name, feature_set_version=feature_set_version).set(
        value
    )


def set_feature_std(feature_name: str, feature_set_version: str, value: float) -> None:
    feature_std.labels(feature_name=feature_name, feature_set_version=feature_set_version).set(
        max(0.0, value)
    )


def record_feature_nan(feature_name: str, feature_set_version: str) -> None:
    feature_nan_total.labels(
        feature_name=feature_name, feature_set_version=feature_set_version
    ).inc()
