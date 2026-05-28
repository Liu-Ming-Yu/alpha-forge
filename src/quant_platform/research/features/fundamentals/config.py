"""Configuration for the ``fundamentals-plus-v1`` feature set."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.research.features.contracts import BaseFamilyConfig

#: Feature-set version string. Bumped from the legacy
#: ``"fundamentals-starter-v1"`` (9 features in
#: ``research/fundamentals/features.py``) when the catalog expanded to
#: ~40 features and started carrying the new
#: :class:`FeatureSpec` metadata (``signal_timestamp``,
#: ``canonical_name``, ``aliases``).
FEATURE_SET_VERSION: str = "fundamentals-plus-v1"

#: Window sizes for the rolling fundamental aggregates. TTM = trailing
#: 4 quarters. YoY = compare to the same fiscal quarter one year ago
#: (shift 4 quarters). QoQ = previous fiscal quarter (shift 1).
TTM_WINDOW_QUARTERS: int = 4
YOY_LAG_QUARTERS: int = 4
QOQ_LAG_QUARTERS: int = 1

#: The legacy 9-feature catalog version. Surfaced here so the compat
#: shim in ``research/fundamentals/features.py`` can pin its
#: ``FEATURE_NAMES`` selection without re-importing from this module
#: at runtime.
LEGACY_VERSION: str = "fundamentals-starter-v1"


@dataclass(frozen=True)
class FundamentalsConfig(BaseFamilyConfig):
    """Frozen config for the fundamentals-plus feature factory.

    Attributes
    ----------
    version:
        Feature-set version pinned into the :class:`FeatureSpec` for
        every produced feature. Defaults to :data:`FEATURE_SET_VERSION`.
    require_full_ttm:
        When ``True`` (default), TTM aggregates require all 4 quarters
        present (``min_periods=4``). When ``False``, partial TTMs are
        emitted from the first observation — useful only for diagnostic
        runs against IPOs where 4 quarters of history don't yet exist.
    require_full_yoy:
        When ``True`` (default), YoY ratios require a non-null lag-4
        denominator; the first 4 quarters of every instrument are NaN.
        Mirrors :attr:`require_full_ttm` for the growth/acceleration
        families.
    """

    version: str = FEATURE_SET_VERSION
    require_full_ttm: bool = True
    require_full_yoy: bool = True


DEFAULT_CONFIG: FundamentalsConfig = FundamentalsConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "FEATURE_SET_VERSION",
    "FundamentalsConfig",
    "LEGACY_VERSION",
    "QOQ_LAG_QUARTERS",
    "TTM_WINDOW_QUARTERS",
    "YOY_LAG_QUARTERS",
]
