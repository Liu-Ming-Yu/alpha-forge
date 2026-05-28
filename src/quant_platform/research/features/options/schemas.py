"""Input record schema for the ``options-v1`` feature family.

The platform doesn't yet have an options-chain data feed (CBOE,
OptionMetrics, Polygon options, ORATS — all paid vendor products).
The family is scaffolded around a single :class:`OptionsSnapshot`
dataclass that the operator can populate from any vendor in a
follow-up PR. Tests use synthetic fixtures only.

Why one record per (instrument, date) instead of per-contract:
v1 only needs derived metrics (ATM IV, 25Δ skew, term slope, put/call
volume, OI, realized-vol comparison). Vendors typically publish these
as already-derived "interpolated surface" metrics; reproducing them
from raw chains is a parallel piece of work that lives in a different
PR. v1's contract takes the derived metrics directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class OptionsSnapshot:
    """One daily snapshot of derived options metrics for one instrument.

    Attributes
    ----------
    instrument_id:
        Underlying instrument.
    snapshot_date:
        Date the snapshot is as-of. Tz-aware (UTC recommended).
    iv_30d_atm:
        At-the-money implied volatility at a 30-calendar-day expiry.
        ``None`` allowed when the vendor couldn't fit the surface
        (illiquid name, missing strikes). NaN-safe downstream.
    iv_60d_atm:
        Same at 60-day expiry. Used to compute the term-slope feature.
    iv_25d_call:
        25-delta call IV (out-of-the-money call). Used for skew.
    iv_25d_put:
        25-delta put IV (out-of-the-money put). Used for skew.
    put_volume:
        Total daily option volume across put strikes/expiries.
        Non-negative integer.
    call_volume:
        Same for calls.
    put_open_interest:
        Total open interest across put strikes/expiries. Non-negative.
    call_open_interest:
        Same for calls.
    realized_vol_21d:
        Trailing 21-trading-day realized volatility of daily log
        returns on the underlying. Used for the IV-vs-realized
        premium feature. ``None`` allowed during warm-up.
    """

    instrument_id: str
    snapshot_date: datetime
    iv_30d_atm: float | None
    iv_60d_atm: float | None
    iv_25d_call: float | None
    iv_25d_put: float | None
    put_volume: int
    call_volume: int
    put_open_interest: int
    call_open_interest: int
    realized_vol_21d: float | None

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("OptionsSnapshot.instrument_id must be non-empty")
        if self.snapshot_date.tzinfo is None:
            raise ValueError("OptionsSnapshot.snapshot_date must be timezone-aware")
        for field_name in (
            "iv_30d_atm",
            "iv_60d_atm",
            "iv_25d_call",
            "iv_25d_put",
            "realized_vol_21d",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"OptionsSnapshot.{field_name} must be >= 0 or None; got {value}")
        for field_name in (
            "put_volume",
            "call_volume",
            "put_open_interest",
            "call_open_interest",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"OptionsSnapshot.{field_name} must be >= 0; got {value}")


__all__ = ["OptionsSnapshot"]
