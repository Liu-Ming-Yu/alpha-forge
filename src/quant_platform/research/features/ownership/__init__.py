"""``ownership-v1`` feature family.

Six features derived from 13F institutional holdings + FINRA
short-interest snapshots + shares-outstanding records.

The real 13F and short-interest data feeds are **not yet wired** into
the platform (both are paid vendor products — Sharadar SF3, a 13F
aggregator, or FINRA short-interest files). v1 ships the family
scaffold against explicit input dataclass contracts
(:class:`Holding13FRecord`, :class:`ShortInterestRecord`,
:class:`SharesOutstandingRecord`) so the operator can populate them
from any vendor in a separate (out-of-scope-for-v1) PR. Tests use
synthetic fixtures.

The six features:

* ``institutional_ownership_pct`` — total 13F shares / shares-out.
* ``institutional_holder_count`` — distinct 13F filers.
* ``institutional_ownership_change_63d`` — quarter-over-quarter diff.
* ``short_interest_ratio`` — short-interest / shares-out.
* ``days_to_cover`` — short-interest / avg-daily-volume.
* ``short_interest_change_20d`` — multi-week diff in the ratio.

All six ship ``expected_direction="unknown"``,
``larger_is_better=False`` — evidence-gated.

See :mod:`quant_platform.research.features.ownership.features` for
the full catalogue and compute pipeline.
"""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.ownership.config import (
    DEFAULT_13F_AVAILABILITY_LAG_DAYS,
    DEFAULT_13F_CHANGE_WINDOW_DAYS,
    DEFAULT_CONFIG,
    DEFAULT_SHORT_INTEREST_AVAILABILITY_LAG_DAYS,
    DEFAULT_SHORT_INTEREST_CHANGE_WINDOW_DAYS,
    FEATURE_SET_VERSION,
    OwnershipConfig,
)
from quant_platform.research.features.ownership.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_INPUT_COLUMNS,
    compute_ownership_features,
)
from quant_platform.research.features.ownership.schemas import (
    Holding13FRecord,
    SharesOutstandingRecord,
    ShortInterestRecord,
)
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

MANIFEST: FamilyManifest = FamilyManifest(
    name="ownership",
    version=FEATURE_SET_VERSION,
    feature_specs=FEATURE_SPECS,
    required_input_columns=REQUIRED_INPUT_COLUMNS,
    key_columns=DEFAULT_KEY_COLUMNS,
    default_training_feature_names=DEFAULT_TRAINING_FEATURE_NAMES,
)

# Side-effect: importing this package registers the manifest into the
# process-global registry. ``register_family`` is idempotent.
register_family(MANIFEST)


__all__ = [
    "DEFAULT_13F_AVAILABILITY_LAG_DAYS",
    "DEFAULT_13F_CHANGE_WINDOW_DAYS",
    "DEFAULT_CONFIG",
    "DEFAULT_SHORT_INTEREST_AVAILABILITY_LAG_DAYS",
    "DEFAULT_SHORT_INTEREST_CHANGE_WINDOW_DAYS",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "Holding13FRecord",
    "MANIFEST",
    "OwnershipConfig",
    "REQUIRED_INPUT_COLUMNS",
    "SharesOutstandingRecord",
    "ShortInterestRecord",
    "compute_ownership_features",
]
