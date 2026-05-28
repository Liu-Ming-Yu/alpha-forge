"""Input record schemas for the ``ownership-v1`` feature family.

The platform doesn't yet have a wired 13F or short-interest data feed
(both are paid vendor products — Sharadar SF3, FINRA short-interest
files, or a 13F aggregator). The family is therefore scaffolded around
explicit dataclass contracts that downstream code can populate from any
vendor: synthetic data in tests today, a real feed in a follow-up
that's out of scope for v1.

Three input records:

* :class:`Holding13FRecord` — one line of one institutional holder's
  quarterly 13F filing. Indexed by ``(filer_id, instrument_id,
  period_end)``.
* :class:`ShortInterestRecord` — one bi-monthly FINRA short-interest
  snapshot for one instrument.
* :class:`SharesOutstandingRecord` — periodic shares-outstanding
  snapshot. Needed to normalise both 13F shares and short-interest
  shares into percentages of float. Fundamentals families already
  carry this in Sharadar SF1; v1 takes it as an explicit input so the
  family doesn't depend on a sibling family at compute time.

Stability contract: every schema is frozen + range-validated in
``__post_init__``. Adding/renaming a field is a v2 bump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class Holding13FRecord:
    """One 13F holding line.

    Attributes
    ----------
    filer_id:
        Stable identifier for the institutional holder (CIK,
        FactSet manager ID, etc.). Used to count distinct holders
        and compute concentration.
    instrument_id:
        The held instrument.
    period_end:
        Quarter-end date the holdings are as-of (typically
        ``YYYY-03-31``, ``-06-30``, ``-09-30``, ``-12-31``). 13F
        filings are due 45 days after period_end, so the panel must
        not surface the row before ``period_end + 45d`` — PIT-safety
        is the caller's responsibility (the aggregator respects
        ``available_at`` if supplied).
    shares_held:
        Number of shares held at period_end. Non-negative.
    market_value:
        USD value at period_end. Non-negative. Used for sanity
        checks but not directly by v1 features.
    available_at:
        Operator-supplied "earliest date the row should enter the
        panel". Defaults to ``period_end + 45 days`` for PIT-safety
        if not provided; the aggregator masks the row from earlier
        dates. The 45-day default matches the SEC's 13F filing
        deadline.
    """

    filer_id: str
    instrument_id: str
    period_end: datetime
    shares_held: int
    market_value: float
    available_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.filer_id.strip():
            raise ValueError("Holding13FRecord.filer_id must be non-empty")
        if not self.instrument_id.strip():
            raise ValueError("Holding13FRecord.instrument_id must be non-empty")
        if self.period_end.tzinfo is None:
            raise ValueError("Holding13FRecord.period_end must be timezone-aware")
        if self.shares_held < 0:
            raise ValueError(f"Holding13FRecord.shares_held must be >= 0; got {self.shares_held}")
        if self.market_value < 0:
            raise ValueError(f"Holding13FRecord.market_value must be >= 0; got {self.market_value}")


@dataclass(frozen=True)
class ShortInterestRecord:
    """One short-interest snapshot.

    Attributes
    ----------
    instrument_id:
        The instrument the short interest is on.
    settlement_date:
        FINRA settlement date the snapshot represents. FINRA
        publishes bi-monthly (mid-month and end-of-month
        settlements).
    short_interest_shares:
        Number of shares sold short and not yet covered.
        Non-negative.
    avg_daily_volume_shares:
        Trailing average daily volume in shares. Used to derive
        days-to-cover. Strictly positive (zero would divide-by-zero).
    available_at:
        Defaults to ``settlement_date + 8 days`` for PIT-safety
        (FINRA's typical publication lag). Override to match the
        operator's actual receipt delay.
    """

    instrument_id: str
    settlement_date: datetime
    short_interest_shares: int
    avg_daily_volume_shares: float
    available_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("ShortInterestRecord.instrument_id must be non-empty")
        if self.settlement_date.tzinfo is None:
            raise ValueError("ShortInterestRecord.settlement_date must be timezone-aware")
        if self.short_interest_shares < 0:
            raise ValueError(
                f"ShortInterestRecord.short_interest_shares must be >= 0; "
                f"got {self.short_interest_shares}"
            )
        if self.avg_daily_volume_shares <= 0:
            raise ValueError(
                f"ShortInterestRecord.avg_daily_volume_shares must be > 0; "
                f"got {self.avg_daily_volume_shares}"
            )


@dataclass(frozen=True)
class SharesOutstandingRecord:
    """Periodic shares-outstanding snapshot.

    Attributes
    ----------
    instrument_id:
        Instrument.
    period_end:
        Date the snapshot is as-of (typically the same period_end
        as a 13F filing or a fundamentals filing).
    shares_outstanding:
        Total shares outstanding. Strictly positive (zero would
        divide-by-zero downstream).
    """

    instrument_id: str
    period_end: datetime
    shares_outstanding: int

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("SharesOutstandingRecord.instrument_id must be non-empty")
        if self.period_end.tzinfo is None:
            raise ValueError("SharesOutstandingRecord.period_end must be timezone-aware")
        if self.shares_outstanding <= 0:
            raise ValueError(
                f"SharesOutstandingRecord.shares_outstanding must be > 0; "
                f"got {self.shares_outstanding}"
            )


__all__ = [
    "Holding13FRecord",
    "ShortInterestRecord",
    "SharesOutstandingRecord",
]
