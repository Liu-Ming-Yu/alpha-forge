"""Prometheus metric catalogue for the platform metrics adapter."""

from __future__ import annotations

import contextlib
from typing import Any, cast

from quant_platform.infrastructure.metrics.registry import (
    PROM_AVAILABLE,
    MetricLike,
    counter,
    gauge,
    get_registry,
    histogram,
    reset_registry,
)

reset_registry()

order_submitted_total: MetricLike = counter(
    "quant_orders_submitted_total",
    "Orders successfully acknowledged by the broker.",
    ("engine",),
)
order_rejected_total: MetricLike = counter(
    "quant_orders_rejected_total",
    "Orders rejected by the pre-trade gate or broker.",
    ("engine", "reason"),
)
fill_latency_seconds: MetricLike = histogram(
    "quant_fill_latency_seconds",
    "Latency from order submission to first fill, in seconds.",
    ("engine",),
    buckets=(0.05, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
)
order_submit_latency_seconds: MetricLike = histogram(
    "quant_order_submit_latency_seconds",
    "Wall-time from SubmitOrdersController.submit() to broker ack, in seconds.",
    ("engine", "outcome"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
cycle_duration_seconds: MetricLike = histogram(
    "quant_cycle_duration_seconds",
    "Wall-time per cycle phase (signals, regime, target, gate, submit, drain, total).",
    ("engine", "phase"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
cycle_errors_total: MetricLike = counter(
    "quant_cycle_errors_total",
    "Unhandled exceptions that escaped a strategy cycle, by engine.",
    ("engine",),
)
cash_drift_usd: MetricLike = gauge(
    "quant_cash_drift_usd",
    "Absolute cash drift observed during reconciliation (USD).",
    ("engine",),
)
reconciliation_mismatches_total: MetricLike = counter(
    "quant_reconciliation_mismatches_total",
    "Number of reconciliation mismatches detected, by type.",
    ("engine", "type"),
)
lock_lease_lost_total: MetricLike = counter(
    "quant_lock_lease_lost_total",
    "Distributed-lock lease losses (monotonic).",
    ("resource",),
)
lock_operations_total: MetricLike = counter(
    "quant_lock_operations_total",
    "Distributed-lock operation counts.",
    ("resource", "op", "outcome"),
)
event_bus_publish_total: MetricLike = counter(
    "quant_event_bus_publish_total",
    "Events published, by backend and outcome.",
    ("backend", "outcome"),
)
event_bus_stream_length: MetricLike = gauge(
    "quant_event_bus_stream_length",
    "Total entries in the Redis Stream (XLEN).",
    ("backend", "stream"),
)
event_bus_pending_entries_total: MetricLike = gauge(
    "quant_event_bus_pending_entries_total",
    "Pending entries per consumer group (XPENDING); authoritative PEL depth.",
    ("backend", "stream", "group"),
)
throttle_tokens: MetricLike = gauge(
    "quant_throttle_tokens",
    "Current tokens remaining in the OrderThrottle bucket.",
)
kill_switch_active: MetricLike = gauge(
    "quant_kill_switch_active",
    "1 when any kill-switch is active, else 0.",
)
TEXT_SIGNAL_IC_ROLLING_20D: MetricLike = gauge(
    "quant_text_signal_ic_rolling_20d",
    "Rolling 20-day Spearman IC of the shadow text signal (NaN until 20 observations).",
)
alpha_source_score_coverage: MetricLike = gauge(
    "quant_alpha_source_score_coverage",
    "Fraction of instruments scored by an alpha source in the current cycle.",
    ("source", "mode"),
)
alpha_source_errors_total: MetricLike = counter(
    "quant_alpha_source_errors_total",
    "Alpha source scoring/extraction/ingestion failures.",
    ("source", "error_type"),
)
alpha_text_extraction_latency_seconds: MetricLike = histogram(
    "quant_alpha_text_extraction_latency_seconds",
    "LLM text extraction latency in seconds.",
    ("model", "outcome"),
    buckets=(0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)
alpha_text_ingestion_lag_seconds: MetricLike = gauge(
    "quant_alpha_text_ingestion_lag_seconds",
    "Lag from source publication to text-event ingestion.",
    ("provider",),
)
alpha_ensemble_live_cap: MetricLike = gauge(
    "quant_alpha_ensemble_live_cap",
    "Current cap for non-classical alpha contribution in live ensemble mode.",
)
alpha_live_ramp_level: MetricLike = gauge(
    "quant_alpha_live_ramp_level",
    "Current live ramp level for promoted alpha sources.",
)
http_requests_total: MetricLike = counter(
    "quant_http_requests_total",
    "Total HTTP requests handled by the operator API.",
    ("method", "endpoint", "status"),
)
http_request_duration_seconds: MetricLike = histogram(
    "quant_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ("method", "endpoint"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
event_bus_dead_letter_total: MetricLike = counter(
    "quant_event_bus_dead_letter_total",
    "Entries moved to ``<stream>.dlq`` after exceeding the retry budget.",
    ("backend", "stream", "group"),
)
event_bus_dead_letter_depth: MetricLike = gauge(
    "quant_event_bus_dead_letter_depth",
    "Current depth of ``<stream>.dlq`` as reported by XLEN.",
    ("backend", "stream"),
)

db_query_latency_seconds: MetricLike = histogram(
    "quant_db_query_latency_seconds",
    "Latency of PostgreSQL repository operations, in seconds.",
    ("operation",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
db_pool_utilization: MetricLike = gauge(
    "quant_db_pool_utilization",
    "SQLAlchemy connection pool utilisation by state (checked_out / idle / overflow).",
    ("state",),
)
broker_operation_latency_seconds: MetricLike = histogram(
    "quant_broker_operation_latency_seconds",
    "Latency of broker gateway operations.",
    ("operation",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
risk_gate_rejections_total: MetricLike = counter(
    "quant_risk_gate_rejections_total",
    "Pre-trade gate rejection count by reason.",
    ("reason",),
)
cash_reservation_age_seconds: MetricLike = gauge(
    "quant_cash_reservation_age_seconds",
    "Age of the oldest active cash reservation in seconds.",
)
order_ack_latency_seconds: MetricLike = histogram(
    "quant_order_ack_latency_seconds",
    "Latency from order placement to broker acknowledgement, in seconds.",
    ("engine",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
bar_read_latency_seconds: MetricLike = histogram(
    "quant_bar_read_latency_seconds",
    "Latency of bar store read operations, in seconds.",
    ("operation",),
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)
feature_compute_latency_seconds: MetricLike = histogram(
    "quant_feature_compute_latency_seconds",
    "Latency of feature computation per alpha source, in seconds.",
    ("source",),
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
)
realized_slippage_bps: MetricLike = histogram(
    "quant_realized_slippage_bps",
    "Realized fill slippage in basis points vs. limit/model price.",
    ("engine",),
    buckets=(-50.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 50.0, 100.0),
)
fill_rate_pct: MetricLike = gauge(
    "quant_fill_rate_pct",
    "Fraction of submitted orders fully filled in the current cycle.",
    ("engine",),
)
bar_gaps_detected_total: MetricLike = counter(
    "quant_bar_gaps_detected_total",
    "Number of bar continuity gaps detected during ingest, by instrument.",
    ("instrument_id",),
)
event_deser_errors_total: MetricLike = counter(
    "quant_event_deser_errors_total",
    "Event deserialisation failures, by event_type.",
    ("event_type",),
)
pg_pool_checked_out: MetricLike = gauge(
    "quant_pg_pool_checked_out",
    "Number of PostgreSQL connections currently checked out from the pool.",
)
redis_connection_errors_total: MetricLike = counter(
    "quant_redis_connection_errors_total",
    "Redis connection-level errors.",
)
feature_mean: MetricLike = gauge(
    "quant_feature_mean",
    "Current mean value per feature.",
    ("feature_name", "feature_set_version"),
)
feature_std: MetricLike = gauge(
    "quant_feature_std",
    "Current standard deviation per feature.",
    ("feature_name", "feature_set_version"),
)
feature_nan_total: MetricLike = counter(
    "quant_feature_nan_total",
    "Cumulative count of NaN/inf values encountered per feature.",
    ("feature_name", "feature_set_version"),
)

if PROM_AVAILABLE:
    with contextlib.suppress(Exception):
        import prometheus_client

        registry = get_registry()
        if registry is not None:
            prometheus_client.ProcessCollector(registry=cast("Any", registry))
            prometheus_client.PlatformCollector(registry=cast("Any", registry))
