"""``learned-representations-v1`` feature family.

Artifact-backed PCA representation family — the **first family in the
catalogue that's representation-only**. It transforms features
emitted by the other 9 families through a frozen
:class:`PCAArtifact`, producing 8 principal-component scores plus a
reconstruction error. **No fitting happens inside this family at
compute time** — the artifact is trained out-of-band by
:func:`~.trainer.fit_pca_artifact`, persisted as JSON via
:func:`~.loader.save_pca_artifact`, and loaded by the operator
before compute.

The 9 features:

* ``learned_pc_1`` .. ``learned_pc_8`` — PCA scores.
* ``learned_reconstruction_error`` — per-row L2 residual norm.

All evidence-gated (``expected_direction="unknown"``,
``larger_is_better=False``). Promotion is a family-version bump.

See :doc:`docs/learned-representations-v1-family.md` for the full
reference and :doc:`docs/adr-002-learned-family-representation-choice.md`
for why PCA was selected over autoencoders, MoE, XGBoost leaf
indices, and explicit interaction expansion.
"""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.learned.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    PCAArtifact,
)
from quant_platform.research.features.learned.config import (
    DEFAULT_CONFIG,
    DEFAULT_N_COMPONENTS,
    FEATURE_SET_VERSION,
    LearnedConfig,
)
from quant_platform.research.features.learned.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_INPUT_COLUMNS,
    compute_learned_features,
)
from quant_platform.research.features.learned.loader import (
    load_pca_artifact,
    save_pca_artifact,
)
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

MANIFEST: FamilyManifest = FamilyManifest(
    name="learned",
    version=FEATURE_SET_VERSION,
    feature_specs=FEATURE_SPECS,
    required_input_columns=REQUIRED_INPUT_COLUMNS,
    key_columns=DEFAULT_KEY_COLUMNS,
    default_training_feature_names=DEFAULT_TRAINING_FEATURE_NAMES,
)

register_family(MANIFEST)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_CONFIG",
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "LearnedConfig",
    "MANIFEST",
    "PCAArtifact",
    "REQUIRED_INPUT_COLUMNS",
    "compute_learned_features",
    "load_pca_artifact",
    "save_pca_artifact",
]
