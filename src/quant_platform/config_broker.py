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
    # Mode → port mapping for data sync. IBKR conventions: TWS paper 7497 /
    # live 7496; IB Gateway paper 4002 / live 4001. ``resolved_port(mode)``
    # selects the paper port in paper/shadow mode and the live port in live
    # mode (TWS unless ``use_gateway``), honoring an explicitly-pinned ``port``
    # when it already belongs to the requested mode's family.
    paper_port: int = 7497
    live_port: int = 7496
    gateway_paper_port: int = 4002
    gateway_live_port: int = 4001
    use_gateway: bool = False
    read_only_client_id: int = Field(
        default=0,
        description=(
            "Dedicated client id for read-only TWS data syncs so they never "
            "collide with a live trading connection. 0 (default) derives "
            "client_id + 90."
        ),
    )
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

    @staticmethod
    def _is_live(mode: str) -> bool:
        return mode.strip().lower() == "live"

    def resolved_host(self) -> str:
        return self.host.strip() or "127.0.0.1"

    def resolved_port(self, mode: str) -> int:
        """Return the TWS/Gateway port for ``mode`` (paper/shadow → paper port,
        live → live port). IB Gateway is used when ``use_gateway`` is set or the
        configured ``port`` is already a Gateway port; otherwise TWS."""
        live = self._is_live(mode)
        gateway = self.use_gateway or self.port in {
            self.gateway_paper_port,
            self.gateway_live_port,
        }
        if gateway:
            return self.gateway_live_port if live else self.gateway_paper_port
        return self.live_port if live else self.paper_port

    def resolved_paper_trading(self, mode: str) -> bool:
        """Paper/shadow modes are non-live; only ``live`` trades real money."""
        return not self._is_live(mode)

    def sync_client_id(self) -> int:
        """Client id for a read-only data sync (never the trading client id)."""
        return self.read_only_client_id or (self.client_id + 90)
