"""``fundamentals-plus-v1`` feature family.

Importing this package registers the family's :class:`FeatureSpec` set
into the process-global registry; downstream code can then look up
specs by name without re-importing this module.
"""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.fundamentals.config import (
    DEFAULT_CONFIG,
    FEATURE_SET_VERSION,
    LEGACY_VERSION,
    FundamentalsConfig,
)
from quant_platform.research.features.fundamentals.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    compute_fundamentals_features,
)
from quant_platform.research.features.fundamentals.panel import (
    REQUIRED_INPUT_COLUMNS,
    REQUIRED_RAW_COLUMNS,  # deprecated alias of REQUIRED_INPUT_COLUMNS
    prepare_fundamentals_panel,
)
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.transforms import KEY_COLUMNS_FUNDAMENTALS

MANIFEST: FamilyManifest = FamilyManifest(
    name="fundamentals",
    version=FEATURE_SET_VERSION,
    feature_specs=FEATURE_SPECS,
    required_input_columns=REQUIRED_INPUT_COLUMNS,
    key_columns=KEY_COLUMNS_FUNDAMENTALS,
    default_training_feature_names=DEFAULT_TRAINING_FEATURE_NAMES,
)

# Side-effect: importing this package registers the manifest (and every
# spec it carries) into the process-global registry. ``register_family``
# is idempotent for identical manifests so re-importing this package is
# a no-op rather than an error.
register_family(MANIFEST)

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "LEGACY_VERSION",
    "MANIFEST",
    "REQUIRED_INPUT_COLUMNS",
    "REQUIRED_RAW_COLUMNS",  # deprecated alias
    "FundamentalsConfig",
    "compute_fundamentals_features",
    "prepare_fundamentals_panel",
]
