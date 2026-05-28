"""Configuration for the ``estimates-v1`` feature set."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.research.features.contracts import BaseFamilyConfig

#: Family version. Bump on formula change, input addition, or column
#: rename. v1 ships 6 features against scaffolded ConsensusSnapshot +
#: EarningsSurpriseRecord input contracts; the real IBES / FactSet /
#: Visible Alpha feed wiring is a separate (deferred) PR.
FEATURE_SET_VERSION: str = "estimates-v1"

#: Default target period for the EPS estimate features. ``FY1`` (next
#: fiscal year) has the deepest analyst coverage and is the IBES-
#: standard reference period for "the consensus".
DEFAULT_EPS_TARGET_PERIOD: str = "FY1"

#: Default target period for the revenue estimate features. Same
#: reasoning as EPS — FY1 is the most-covered period.
DEFAULT_REVENUE_TARGET_PERIOD: str = "FY1"

#: Default rolling window (trading days) for the revision-magnitude
#: features (``eps_estimate_revision_30d``,
#: ``revenue_estimate_revision_30d``). 30 calendar days is the IBES
#: convention for "recent revisions"; on a trading-day panel this
#: maps to ~21 trading days but the column name uses the calendar
#: convention for operator readability.
DEFAULT_REVISION_WINDOW_DAYS: int = 30

#: Default number of trailing fiscal periods to average for the
#: surprise feature. 4 quarters = 1 fiscal year of surprise history.
DEFAULT_SURPRISE_LOOKBACK_QUARTERS: int = 4


@dataclass(frozen=True)
class EstimatesConfig(BaseFamilyConfig):
    """Frozen config for the estimate-revisions feature factory.

    Attributes
    ----------
    version:
        Family-version string stamped into every emitted
        :class:`FeatureSpec`. Defaults to :data:`FEATURE_SET_VERSION`.
    eps_target_period:
        Which fiscal period the EPS estimate features track (default
        ``"FY1"`` — next fiscal year). Must be one of
        :data:`~.schemas.ALLOWED_TARGET_PERIODS`.
    revenue_target_period:
        Same for revenue estimates.
    revision_window_days:
        Calendar-day lookback for the revision features. Appears in
        feature column names, so changing it requires a family-version
        bump.
    surprise_lookback_quarters:
        Number of trailing fiscal periods to average for the
        ``eps_surprise_mean_4q`` feature. The integer appears in the
        column name; same versioning rule.
    """

    version: str = FEATURE_SET_VERSION
    eps_target_period: str = DEFAULT_EPS_TARGET_PERIOD
    revenue_target_period: str = DEFAULT_REVENUE_TARGET_PERIOD
    revision_window_days: int = DEFAULT_REVISION_WINDOW_DAYS
    surprise_lookback_quarters: int = DEFAULT_SURPRISE_LOOKBACK_QUARTERS

    def __post_init__(self) -> None:
        super().__post_init__()
        # Lazy import to avoid a circular at module load (schemas imports
        # nothing from this module, but keeping it lazy is harmless).
        from quant_platform.research.features.estimates.schemas import (
            ALLOWED_TARGET_PERIODS,
        )

        if self.eps_target_period not in ALLOWED_TARGET_PERIODS:
            raise ValueError(
                f"EstimatesConfig.eps_target_period must be one of "
                f"{ALLOWED_TARGET_PERIODS!r}; got {self.eps_target_period!r}"
            )
        if self.revenue_target_period not in ALLOWED_TARGET_PERIODS:
            raise ValueError(
                f"EstimatesConfig.revenue_target_period must be one of "
                f"{ALLOWED_TARGET_PERIODS!r}; got {self.revenue_target_period!r}"
            )
        if self.revision_window_days < 1:
            raise ValueError("EstimatesConfig.revision_window_days must be >= 1")
        if self.surprise_lookback_quarters < 1:
            raise ValueError("EstimatesConfig.surprise_lookback_quarters must be >= 1")


DEFAULT_CONFIG: EstimatesConfig = EstimatesConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_EPS_TARGET_PERIOD",
    "DEFAULT_REVENUE_TARGET_PERIOD",
    "DEFAULT_REVISION_WINDOW_DAYS",
    "DEFAULT_SURPRISE_LOOKBACK_QUARTERS",
    "EstimatesConfig",
    "FEATURE_SET_VERSION",
]
