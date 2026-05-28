"""Input record schema for the ``macro-v1`` feature family.

The platform doesn't have an internal macro-data feed; v1 takes one
flat input contract — a stream of :class:`MacroSeriesValue`
observations — that the operator can populate from any source (FRED
via the public free API, Sharadar Macro, Quandl, Bloomberg, or a
hand-curated CSV).

A separate operator-only helper at :mod:`.fetcher` wraps the FRED
API for the common case. It lazy-imports ``fredapi`` so the family
itself stays light-dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class MacroSeriesValue:
    """One daily observation of one macro time series.

    Attributes
    ----------
    series_id:
        Stable identifier for the series. v1 uses FRED series IDs
        (e.g. ``"DGS10"`` for the 10-year Treasury yield) but any
        stable string works.
    observation_date:
        The date the value is as-of. FRED observations are typically
        published at end-of-day for the named ``observation_date``,
        but the family treats the observation date itself as the
        public-availability date. Tz-aware.
    value:
        The observed value. Series-specific units (percent for
        yields, level for VIX, index level for dollar index).
        ``None``-like missing values must be filtered upstream — the
        schema requires a finite float.
    """

    series_id: str
    observation_date: datetime
    value: float

    def __post_init__(self) -> None:
        if not self.series_id.strip():
            raise ValueError("MacroSeriesValue.series_id must be non-empty")
        if self.observation_date.tzinfo is None:
            raise ValueError("MacroSeriesValue.observation_date must be timezone-aware")
        # Reject NaN / inf at the boundary — downstream features
        # assume finite floats and a NaN would silently propagate.
        if not (self.value == self.value):  # NaN check
            raise ValueError(
                f"MacroSeriesValue.value must be a finite float; got NaN for "
                f"series_id={self.series_id!r}"
            )


__all__ = ["MacroSeriesValue"]
