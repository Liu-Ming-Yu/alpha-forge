"""``estimates-v1`` feature family.

Six features derived from analyst-consensus snapshots + historical
earnings-surprise records.

The real IBES / FactSet / Visible Alpha data feeds are **not yet
wired** into the platform (all paid vendor products). v1 ships the
family scaffold against explicit input dataclass contracts
(:class:`ConsensusSnapshot`, :class:`EarningsSurpriseRecord`) so the
operator can populate them from any vendor in a separate
(out-of-scope-for-v1) PR. Tests use synthetic fixtures.

The six features:

* ``eps_estimate_revision_30d`` — relative drift in FY1 EPS consensus.
* ``eps_estimate_up_vs_down_30d`` — directional revision balance.
* ``eps_estimate_dispersion`` — cross-analyst uncertainty (std/mean).
* ``analyst_coverage_count`` — distinct analysts covering.
* ``eps_surprise_mean_4q`` — past-4-quarter average % surprise.
* ``revenue_estimate_revision_30d`` — relative drift in FY1 revenue
  consensus.

All six ship ``expected_direction="unknown"``,
``larger_is_better=False`` — evidence-gated.

See :mod:`quant_platform.research.features.estimates.features` for the
full catalogue and compute pipeline.
"""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.estimates.config import (
    DEFAULT_CONFIG,
    DEFAULT_EPS_TARGET_PERIOD,
    DEFAULT_REVENUE_TARGET_PERIOD,
    DEFAULT_REVISION_WINDOW_DAYS,
    DEFAULT_SURPRISE_LOOKBACK_QUARTERS,
    FEATURE_SET_VERSION,
    EstimatesConfig,
)
from quant_platform.research.features.estimates.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_INPUT_COLUMNS,
    compute_estimate_features,
)
from quant_platform.research.features.estimates.schemas import (
    ALLOWED_ESTIMATE_KINDS,
    ALLOWED_TARGET_PERIODS,
    ConsensusSnapshot,
    EarningsSurpriseRecord,
)
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

MANIFEST: FamilyManifest = FamilyManifest(
    name="estimates",
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
    "ALLOWED_ESTIMATE_KINDS",
    "ALLOWED_TARGET_PERIODS",
    "ConsensusSnapshot",
    "DEFAULT_CONFIG",
    "DEFAULT_EPS_TARGET_PERIOD",
    "DEFAULT_REVENUE_TARGET_PERIOD",
    "DEFAULT_REVISION_WINDOW_DAYS",
    "DEFAULT_SURPRISE_LOOKBACK_QUARTERS",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "EarningsSurpriseRecord",
    "EstimatesConfig",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "MANIFEST",
    "REQUIRED_INPUT_COLUMNS",
    "compute_estimate_features",
]
