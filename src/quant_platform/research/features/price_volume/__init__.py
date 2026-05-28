"""``price-volume-starter-v1`` feature family."""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.price_volume.config import (
    DEFAULT_CONFIG,
    FEATURE_SET_VERSION,
    PriceVolumeConfig,
)
from quant_platform.research.features.price_volume.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_BAR_COLUMNS,  # deprecated alias of REQUIRED_INPUT_COLUMNS
    REQUIRED_INPUT_COLUMNS,
    compute_price_volume_features,
)
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

MANIFEST: FamilyManifest = FamilyManifest(
    name="price_volume",
    version=FEATURE_SET_VERSION,
    feature_specs=FEATURE_SPECS,
    required_input_columns=REQUIRED_INPUT_COLUMNS,
    key_columns=DEFAULT_KEY_COLUMNS,
    default_training_feature_names=DEFAULT_TRAINING_FEATURE_NAMES,
)

# Side-effect: importing this package registers the manifest (and every
# spec it carries) into the process-global registry.
register_family(MANIFEST)

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "MANIFEST",
    "PriceVolumeConfig",
    "REQUIRED_BAR_COLUMNS",  # deprecated alias
    "REQUIRED_INPUT_COLUMNS",
    "compute_price_volume_features",
]
