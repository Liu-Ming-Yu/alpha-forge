"""``macro-v1`` feature family.

Six macro/regime features derived from 8 FRED time series:
yield-curve slopes (10y-2y, 10y-3m), Baa-Aaa credit spread, VIX level,
30-day dollar-index momentum, and the 10-year TIPS real yield.

Unlike the other families, macro features are **scalar per date** —
the same value gets broadcast across all instruments. The compute
function takes an explicit ``instruments`` list and produces a
standard (instrument_id, date)-keyed :class:`FeatureFrame`.

The family itself is feed-agnostic — it takes a
:class:`MacroSeriesValue` iterable. The operator can populate this
from any source. For the common case of fetching from FRED (free
public API), a thin convenience helper
:func:`~.fetcher.fetch_fred_series` lazy-imports ``fredapi`` and
returns records ready for the compute function.

All six features ship ``expected_direction="unknown"``,
``larger_is_better=False`` — evidence-gated.
"""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.macro.config import (
    DEFAULT_CONFIG,
    DEFAULT_DOLLAR_INDEX_WINDOW_DAYS,
    FEATURE_SET_VERSION,
    FRED_CORPORATE_AAA,
    FRED_CORPORATE_BAA,
    FRED_DOLLAR_INDEX,
    FRED_TIPS_10Y,
    FRED_TREASURY_2Y,
    FRED_TREASURY_3M,
    FRED_TREASURY_10Y,
    FRED_VIX,
    REQUIRED_SERIES_IDS,
    MacroConfig,
)
from quant_platform.research.features.macro.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_INPUT_COLUMNS,
    compute_macro_features,
)
from quant_platform.research.features.macro.schemas import MacroSeriesValue
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

MANIFEST: FamilyManifest = FamilyManifest(
    name="macro",
    version=FEATURE_SET_VERSION,
    feature_specs=FEATURE_SPECS,
    required_input_columns=REQUIRED_INPUT_COLUMNS,
    key_columns=DEFAULT_KEY_COLUMNS,
    default_training_feature_names=DEFAULT_TRAINING_FEATURE_NAMES,
)

register_family(MANIFEST)


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_DOLLAR_INDEX_WINDOW_DAYS",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "FRED_CORPORATE_AAA",
    "FRED_CORPORATE_BAA",
    "FRED_DOLLAR_INDEX",
    "FRED_TIPS_10Y",
    "FRED_TREASURY_10Y",
    "FRED_TREASURY_2Y",
    "FRED_TREASURY_3M",
    "FRED_VIX",
    "MANIFEST",
    "MacroConfig",
    "MacroSeriesValue",
    "REQUIRED_INPUT_COLUMNS",
    "REQUIRED_SERIES_IDS",
    "compute_macro_features",
]
