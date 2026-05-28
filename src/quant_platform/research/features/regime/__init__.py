"""``regime-v1`` feature family.

Produces regime × base-feature interaction columns so the IC-weighted
ranker can rotate weights by market regime. See
[ADR-005](../../../../../docs/architecture/adr-005-regime-overlay-via-interaction-features.md)
for the design framing.

The family consumes the existing :mod:`quant_platform.core.regime`
classifier without modification — research and live share the same
regime detector, so the audit trail can match research evidence to
live execution decisions one-to-one.
"""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.regime.config import (
    BREADTH_SOURCE_ID,
    DEFAULT_CONFIG,
    DEFAULT_INTERACTIONS,
    FEATURE_SET_VERSION,
    INDEX_PROXY_ID,
    REGIME_INDICATOR_LABELS,
    REGIME_STAT_COLUMNS,
    RegimeFeatureConfig,
    RegimeInteractionSpec,
)
from quant_platform.research.features.regime.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_INPUT_COLUMNS,
    compute_regime_features,
    regime_detector_metadata,
)
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

MANIFEST: FamilyManifest = FamilyManifest(
    name="regime",
    version=FEATURE_SET_VERSION,
    feature_specs=FEATURE_SPECS,
    required_input_columns=REQUIRED_INPUT_COLUMNS,
    key_columns=DEFAULT_KEY_COLUMNS,
    default_training_feature_names=DEFAULT_TRAINING_FEATURE_NAMES,
)

# Side-effect: importing this package registers the manifest into the
# process-global registry. Matches the convention used by every other
# feature family.
register_family(MANIFEST)


__all__ = [
    "BREADTH_SOURCE_ID",
    "DEFAULT_CONFIG",
    "DEFAULT_INTERACTIONS",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "INDEX_PROXY_ID",
    "MANIFEST",
    "REGIME_INDICATOR_LABELS",
    "REGIME_STAT_COLUMNS",
    "REQUIRED_INPUT_COLUMNS",
    "RegimeFeatureConfig",
    "RegimeInteractionSpec",
    "compute_regime_features",
    "regime_detector_metadata",
]
