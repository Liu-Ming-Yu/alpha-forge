"""Service-facing metric helpers.

This facade keeps service modules decoupled from the concrete Prometheus
adapter while the platform migrates toward injected telemetry ports/events.
"""

from __future__ import annotations

from quant_platform.infrastructure.metrics import (
    TEXT_SIGNAL_IC_ROLLING_20D,
    observe_alpha_text_extraction_latency,
    observe_fill_latency,
    observe_order_submit_latency,
    record_alpha_source_error,
    record_cycle_error,
    record_feature_nan,
    record_http_request,
    record_order_rejected,
    record_order_submitted,
    record_reconciliation_mismatch,
    render_latest,
    set_alpha_live_controls,
    set_alpha_source_coverage,
    set_alpha_text_ingestion_lag,
    set_cash_drift,
    set_feature_mean,
    set_feature_std,
    set_kill_switch,
    set_throttle_tokens,
    time_phase,
)

__all__ = [
    "TEXT_SIGNAL_IC_ROLLING_20D",
    "observe_alpha_text_extraction_latency",
    "observe_fill_latency",
    "observe_order_submit_latency",
    "record_alpha_source_error",
    "record_cycle_error",
    "record_feature_nan",
    "record_http_request",
    "record_order_rejected",
    "record_order_submitted",
    "record_reconciliation_mismatch",
    "render_latest",
    "set_alpha_live_controls",
    "set_alpha_source_coverage",
    "set_alpha_text_ingestion_lag",
    "set_cash_drift",
    "set_feature_mean",
    "set_feature_std",
    "set_kill_switch",
    "set_throttle_tokens",
    "time_phase",
]
