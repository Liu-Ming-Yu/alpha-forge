"""``options-v1`` feature family.

Six features derived from daily options-implied snapshots: ATM IV at
30-day expiry, 25Δ skew, IV term slope, put/call volume + OI ratios,
and the IV-vs-realized vol premium.

The real options-chain data feed (CBOE, OptionMetrics, Polygon options,
ORATS — all paid vendor products) is **not yet wired** into the
platform. v1 ships the family scaffold against an explicit
:class:`OptionsSnapshot` dataclass contract so the operator can
populate it from any vendor in a separate (out-of-scope-for-v1) PR.
Tests use synthetic fixtures only.

All six features ship ``expected_direction="unknown"``,
``larger_is_better=False`` — evidence-gated.

See :mod:`quant_platform.research.features.options.features` for the
full catalogue and compute pipeline.
"""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.options.config import (
    DEFAULT_ATM_TENOR_DAYS,
    DEFAULT_CONFIG,
    DEFAULT_REALIZED_VOL_WINDOW_DAYS,
    DEFAULT_TERM_LONG_TENOR_DAYS,
    FEATURE_SET_VERSION,
    OptionsConfig,
)
from quant_platform.research.features.options.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_INPUT_COLUMNS,
    compute_options_features,
)
from quant_platform.research.features.options.schemas import OptionsSnapshot
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

MANIFEST: FamilyManifest = FamilyManifest(
    name="options",
    version=FEATURE_SET_VERSION,
    feature_specs=FEATURE_SPECS,
    required_input_columns=REQUIRED_INPUT_COLUMNS,
    key_columns=DEFAULT_KEY_COLUMNS,
    default_training_feature_names=DEFAULT_TRAINING_FEATURE_NAMES,
)

register_family(MANIFEST)


__all__ = [
    "DEFAULT_ATM_TENOR_DAYS",
    "DEFAULT_CONFIG",
    "DEFAULT_REALIZED_VOL_WINDOW_DAYS",
    "DEFAULT_TERM_LONG_TENOR_DAYS",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "MANIFEST",
    "OptionsConfig",
    "OptionsSnapshot",
    "REQUIRED_INPUT_COLUMNS",
    "compute_options_features",
]
