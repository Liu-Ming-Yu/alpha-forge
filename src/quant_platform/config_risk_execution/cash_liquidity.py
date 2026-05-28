"""Cash ledger and liquidity constraint settings."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel


class CashSettings(BaseModel):
    """Cash-ledger reservation and buffer parameters.

    ``buy_side_t1_settlement`` opts the ledger into mirroring sell-side
    T+1/T+2 settlement on the buy side: instead of debiting ``settled_cash``
    at fill time, the buy cost is pushed to the ``unsettled_cash`` pool at
    fill time and moved to ``settled_cash`` on settlement date.  This
    matches the broker's end-of-day cash picture more closely.  Off by
    default until cash-drift telemetry is stable; enable via
    ``QP__CASH__BUY_SIDE_T1_SETTLEMENT=true``.
    """

    reservation_buffer_pct: Decimal = Decimal("0.01")
    reservation_ttl_minutes: int = 30
    drift_tolerance_usd: Decimal = Decimal("1.00")
    buy_side_t1_settlement: bool = False


class LiquiditySettings(BaseModel):
    """Liquidity constraints for pre-trade filtering.

    Used to enforce ADV participation limits: order notional must be less than
    ``adv_participation_pct`` x 20-day average daily volume in USD.

    ``min_adv_usd`` is the minimum acceptable ADV; instruments below this
    threshold are rejected regardless of order size.

    These settings are reserved for the pre-trade gate ADV filter (Phase 4).
    They are defined here so operators can configure them via env vars now.
    """

    adv_participation_pct: float = 0.05  # max 5% of 20-day ADV
    min_adv_usd: float = 1_000_000.0  # min $1M daily volume
    allow_missing_profile: bool = True
