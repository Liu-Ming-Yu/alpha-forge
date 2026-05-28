"""Configuration for the ``ownership-v1`` feature set."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.research.features.contracts import BaseFamilyConfig

#: Family version. Bump on formula change, input addition, or column
#: rename. v1 ships 6 features against scaffolded data contracts; the
#: real 13F + short-interest feed wiring is a separate (deferred) PR.
FEATURE_SET_VERSION: str = "ownership-v1"

#: Default lag (calendar days) between a 13F ``period_end`` and the
#: earliest date the holding may enter the panel. The SEC's 13F filing
#: deadline is 45 days after period_end; the family masks rows from
#: earlier dates to keep the panel point-in-time-safe.
DEFAULT_13F_AVAILABILITY_LAG_DAYS: int = 45

#: Default lag (calendar days) between a FINRA short-interest
#: ``settlement_date`` and the earliest date it may enter the panel.
#: FINRA typically publishes 8 calendar days after settlement.
DEFAULT_SHORT_INTEREST_AVAILABILITY_LAG_DAYS: int = 8

#: Rolling window (trading days) for the multi-period change features
#: (``institutional_ownership_change_qoq``, ``short_interest_change_4w``).
#: Defaults: ~1 quarter for 13F (drift across two filings); ~4 weeks
#: for short interest (drift across two FINRA snapshots).
DEFAULT_13F_CHANGE_WINDOW_DAYS: int = 63  # ~ 1 trading quarter
DEFAULT_SHORT_INTEREST_CHANGE_WINDOW_DAYS: int = 20  # ~ 1 trading month


@dataclass(frozen=True)
class OwnershipConfig(BaseFamilyConfig):
    """Frozen config for the ownership feature factory.

    Attributes
    ----------
    version:
        Family-version string stamped into every emitted
        :class:`FeatureSpec`. Defaults to :data:`FEATURE_SET_VERSION`.
    holding_13f_availability_lag_days:
        PIT-safety lag for 13F records. Overrides
        :class:`Holding13FRecord.available_at` only when that field
        is ``None``.
    short_interest_availability_lag_days:
        PIT-safety lag for short-interest records. Overrides
        :class:`ShortInterestRecord.available_at` only when that
        field is ``None``.
    holding_13f_change_window_days:
        Lookback for the quarter-over-quarter change in institutional
        ownership.
    short_interest_change_window_days:
        Lookback for the multi-week change in short interest.
    """

    version: str = FEATURE_SET_VERSION
    holding_13f_availability_lag_days: int = DEFAULT_13F_AVAILABILITY_LAG_DAYS
    short_interest_availability_lag_days: int = DEFAULT_SHORT_INTEREST_AVAILABILITY_LAG_DAYS
    holding_13f_change_window_days: int = DEFAULT_13F_CHANGE_WINDOW_DAYS
    short_interest_change_window_days: int = DEFAULT_SHORT_INTEREST_CHANGE_WINDOW_DAYS

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.holding_13f_availability_lag_days < 0:
            raise ValueError("OwnershipConfig.holding_13f_availability_lag_days must be >= 0")
        if self.short_interest_availability_lag_days < 0:
            raise ValueError("OwnershipConfig.short_interest_availability_lag_days must be >= 0")
        if self.holding_13f_change_window_days < 1:
            raise ValueError("OwnershipConfig.holding_13f_change_window_days must be >= 1")
        if self.short_interest_change_window_days < 1:
            raise ValueError("OwnershipConfig.short_interest_change_window_days must be >= 1")


DEFAULT_CONFIG: OwnershipConfig = OwnershipConfig()


__all__ = [
    "DEFAULT_13F_AVAILABILITY_LAG_DAYS",
    "DEFAULT_13F_CHANGE_WINDOW_DAYS",
    "DEFAULT_CONFIG",
    "DEFAULT_SHORT_INTEREST_AVAILABILITY_LAG_DAYS",
    "DEFAULT_SHORT_INTEREST_CHANGE_WINDOW_DAYS",
    "FEATURE_SET_VERSION",
    "OwnershipConfig",
]
