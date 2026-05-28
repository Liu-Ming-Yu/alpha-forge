"""Configuration for the ``macro-v1`` feature set."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.research.features.contracts import BaseFamilyConfig

#: Family version. Bump on formula change, input addition, or column
#: rename. v1 ships 6 features driven by 8 FRED series IDs.
FEATURE_SET_VERSION: str = "macro-v1"

# ---------------------------------------------------------------------------
# Required FRED series IDs
# ---------------------------------------------------------------------------
#
# Aliases (operator-facing) map to FRED's canonical IDs so a future
# vendor swap (Sharadar Macro, Quandl) only needs to remap these
# constants. Every required series appears in :data:`REQUIRED_SERIES_IDS`.

#: 10-year Treasury constant maturity yield (percent, daily).
FRED_TREASURY_10Y: str = "DGS10"

#: 2-year Treasury constant maturity yield (percent, daily).
FRED_TREASURY_2Y: str = "DGS2"

#: 3-month Treasury constant maturity yield (percent, daily).
FRED_TREASURY_3M: str = "DGS3MO"

#: Moody's Seasoned Baa Corporate Bond Yield (percent, daily).
FRED_CORPORATE_BAA: str = "BAA"

#: Moody's Seasoned Aaa Corporate Bond Yield (percent, daily).
FRED_CORPORATE_AAA: str = "AAA"

#: CBOE Volatility Index, daily close (percent annualised).
FRED_VIX: str = "VIXCLS"

#: Nominal Broad U.S. Dollar Index (Goods + Services).
FRED_DOLLAR_INDEX: str = "DTWEXBGS"

#: 10-Year Treasury Inflation-Indexed Security Constant Maturity Rate
#: (percent, daily; the real yield curve at the 10-year tenor).
FRED_TIPS_10Y: str = "DFII10"

#: All series IDs the family consumes. The aggregator filters incoming
#: records to this set and rejects unknown series, so a typo in the
#: operator's fetcher surface fails loudly.
REQUIRED_SERIES_IDS: tuple[str, ...] = (
    FRED_TREASURY_10Y,
    FRED_TREASURY_2Y,
    FRED_TREASURY_3M,
    FRED_CORPORATE_BAA,
    FRED_CORPORATE_AAA,
    FRED_VIX,
    FRED_DOLLAR_INDEX,
    FRED_TIPS_10Y,
)


#: Default lookback window (calendar days) for the dollar-index
#: momentum feature. Appears in the column name; changing it requires
#: a family-version bump.
DEFAULT_DOLLAR_INDEX_WINDOW_DAYS: int = 30


@dataclass(frozen=True)
class MacroConfig(BaseFamilyConfig):
    """Frozen config for the macro feature factory.

    Attributes
    ----------
    version:
        Family-version string. Defaults to :data:`FEATURE_SET_VERSION`.
    dollar_index_window_days:
        Calendar-day lookback for the dollar-index percent-change
        feature. Default 30.
    """

    version: str = FEATURE_SET_VERSION
    dollar_index_window_days: int = DEFAULT_DOLLAR_INDEX_WINDOW_DAYS

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dollar_index_window_days < 1:
            raise ValueError("MacroConfig.dollar_index_window_days must be >= 1")


DEFAULT_CONFIG: MacroConfig = MacroConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_DOLLAR_INDEX_WINDOW_DAYS",
    "FEATURE_SET_VERSION",
    "FRED_CORPORATE_AAA",
    "FRED_CORPORATE_BAA",
    "FRED_DOLLAR_INDEX",
    "FRED_TIPS_10Y",
    "FRED_TREASURY_10Y",
    "FRED_TREASURY_2Y",
    "FRED_TREASURY_3M",
    "FRED_VIX",
    "MacroConfig",
    "REQUIRED_SERIES_IDS",
]
