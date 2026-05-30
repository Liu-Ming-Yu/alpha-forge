"""Pure learned-PCA representation compute (inner-layer kernel, ADR-012).

Holds only the artifact, its JSON (de)serialization, the deterministic transform,
and config — no ``register_family`` side-effect and no sklearn trainer. The
research family registration (``research.features.learned``) stays in the
composition layer and imports this compute through shims; the live learned
feature family imports it directly. The sklearn-backed fit
(``research.features.learned.trainer.fit_pca_artifact``) stays in research —
only *inference* is needed in the inner layer.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.learned.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    PCAArtifact,
)
from quant_platform.services.research_service.features.kernel.learned.config import (
    DEFAULT_CONFIG,
    DEFAULT_N_COMPONENTS,
    FEATURE_SET_VERSION,
    LearnedConfig,
)
from quant_platform.services.research_service.features.kernel.learned.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_INPUT_COLUMNS,
    compute_learned_features,
)
from quant_platform.services.research_service.features.kernel.learned.loader import (
    load_pca_artifact,
    save_pca_artifact,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_CONFIG",
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "LearnedConfig",
    "PCAArtifact",
    "REQUIRED_INPUT_COLUMNS",
    "compute_learned_features",
    "load_pca_artifact",
    "save_pca_artifact",
]
