"""Broker connection settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BrokerSettings(BaseModel):
    """IB TWS / Gateway connection parameters.

    ``account_id`` is the IBKR account string (e.g. "DU4502835" for paper).
    ``client_id`` is the numeric API socket identifier (1, 2, 3, ...) used to
    distinguish multiple simultaneous connections; it is NOT the account ID.
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account_id: str = ""
    paper_trading: bool = True
    request_timeout_seconds: float = 10.0
    reconnect_base_delay: float = 2.0
    reconnect_max_delay: float = 60.0
    primary_broker_path: Literal["tws", "client_portal", "dual"] = "tws"
    heartbeat_interval_seconds: float = 5.0
    max_consecutive_health_failures: int = 3
    reconcile_on_reconnect: bool = True
    stale_day_order_cleanup_minutes: int = 30
    stale_gtc_cleanup_minutes: int = Field(
        default=0,
        description=(
            "Cancel GTC/GTD orders that have been open longer than this many minutes. "
            "0 (default) disables GTC cleanup.  DAY orders are controlled separately by "
            "stale_day_order_cleanup_minutes.  "
            "Configured via QP__BROKER__STALE_GTC_CLEANUP_MINUTES."
        ),
    )
    orphan_ttl_minutes: int = Field(
        default=60,
        description=(
            "Submitted orders absent from the broker's open-order list for longer than this "
            "many minutes are treated as orphans: removed from _submitted and a "
            "BrokerOrphanDetected lifecycle event is queued.  0 disables orphan cleanup. "
            "Configured via QP__BROKER__ORPHAN_TTL_MINUTES."
        ),
    )
    historical_bar_fetch_enabled: bool = False
    historical_bar_pacing_window_seconds: float = 600.0
    historical_bar_pacing_max_requests: int = 60
