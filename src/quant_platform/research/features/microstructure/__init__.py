"""``microstructure-v2`` feature family.

Sixteen daily-OHLCV-derived microstructure proxies registered under
``family="microstructure"``. Designed to be complementary to
``price-volume-starter-v1`` — Amihud illiquidity, dollar-volume z-score,
plain high-low range, overnight gap, and the open-to-close return live
in that family. This family ships:

* **Range-based volatility estimators** — Parkinson, Garman-Klass,
  Rogers-Satchell, and Yang-Zhang (drift-independent OHLC+overnight).
* **OHLC-derived bid-ask spread proxies** — Roll (1984),
  Corwin-Schultz (2012).
* **Intraday-position structure** — close-in-range.
* **Serial dependence** — return autocorrelation, volume
  autocorrelation, range persistence.
* **Volume-return coupling**.
* **Range asymmetry**.
* **Jump-robust realised variance** — bipower variation.
* **Higher-moment realised statistics** — realised skewness and
  kurtosis of daily returns.
* **Random-walk test** — Lo-MacKinlay variance ratio VR(5).

Tick/quote-level features (Kyle's λ, VPIN, true effective spread,
order-flow imbalance) defer to a future ``microstructure-v3`` once a
minute-bar or trade-tick feed lands.

All sixteen features ship evidence-gated
(``expected_direction="unknown"``, ``larger_is_better=False``).
Promotion to a directional spec is a family-version bump.

See :mod:`quant_platform.research.features.microstructure.features` for
the full catalogue and compute pipeline.
"""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.microstructure.config import (
    DEFAULT_CONFIG,
    DEFAULT_LONG_WINDOW,
    DEFAULT_SHORT_WINDOW,
    DEFAULT_VARIANCE_RATIO_STRIDE,
    FEATURE_SET_VERSION,
    MicrostructureConfig,
)
from quant_platform.research.features.microstructure.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_INPUT_COLUMNS,
    compute_microstructure_features,
)
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

MANIFEST: FamilyManifest = FamilyManifest(
    name="microstructure",
    version=FEATURE_SET_VERSION,
    feature_specs=FEATURE_SPECS,
    required_input_columns=REQUIRED_INPUT_COLUMNS,
    key_columns=DEFAULT_KEY_COLUMNS,
    default_training_feature_names=DEFAULT_TRAINING_FEATURE_NAMES,
)

# Side-effect: importing this package registers the manifest into the
# process-global registry. ``register_family`` is idempotent for
# identical manifests, so a duplicate import is a no-op.
register_family(MANIFEST)


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_LONG_WINDOW",
    "DEFAULT_SHORT_WINDOW",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "DEFAULT_VARIANCE_RATIO_STRIDE",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "MANIFEST",
    "MicrostructureConfig",
    "REQUIRED_INPUT_COLUMNS",
    "compute_microstructure_features",
]
