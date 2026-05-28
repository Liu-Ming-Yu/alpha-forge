"""Cross-layer financial constants.

These values are domain math, not feature-family configuration — they
belong in ``core`` so every layer (research, services, infrastructure,
edges) can import them without crossing the architecture boundary
enforced by ``scripts/check_import_boundaries.py`` (services may not
import from ``research``).

``features/transforms.py`` re-exports the feature-family-relevant subset
so existing feature code can keep its single import site.
"""

from __future__ import annotations

#: Approximate trading days in one calendar month, used by month-scoped
#: momentum/lookback names (e.g. ``mom_12_1`` = "12 months excluding the
#: most recent 1 month").
TRADING_DAYS_PER_MONTH: int = 21

#: Standard trading-day count in one calendar year, used by Sharpe
#: annualisation (``sqrt(252)``), bucket-Sharpe annualisation
#: (``sqrt(252 / H)``), and any other "annualise from a daily series"
#: math. ``252 = 21 × 12`` is consistent with TRADING_DAYS_PER_MONTH.
TRADING_DAYS_PER_YEAR: int = 252

#: One financial "unit" expressed in basis points. Use when converting
#: between basis points (the wire format for slippage / fees) and
#: decimal returns (``decimal = bps / BPS_PER_UNIT``). Keeps every
#: campaign metrics call on the same conversion.
BPS_PER_UNIT: float = 10_000.0

#: Approximate calendar days in one fiscal quarter. Used by the
#: fundamentals family to convert ``lookback_quarters`` (TTM windows)
#: into the calendar-day ``FeatureSpec.lookback_days`` estimate.
CALENDAR_DAYS_PER_QUARTER: int = 91


__all__ = [
    "BPS_PER_UNIT",
    "CALENDAR_DAYS_PER_QUARTER",
    "TRADING_DAYS_PER_MONTH",
    "TRADING_DAYS_PER_YEAR",
]
