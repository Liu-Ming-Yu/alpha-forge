"""Compatibility shim — learned-PCA feature compute moved to the kernel (ADR-012).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.learned.features``.
This package's ``__init__`` (family registration) re-exports through here; the
live learned family imports the kernel directly.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.learned.features import (
    DEFAULT_TRAINING_FEATURE_NAMES as DEFAULT_TRAINING_FEATURE_NAMES,
)
from quant_platform.services.research_service.features.kernel.learned.features import (
    FEATURE_NAMES as FEATURE_NAMES,
)
from quant_platform.services.research_service.features.kernel.learned.features import (
    FEATURE_SPECS as FEATURE_SPECS,
)
from quant_platform.services.research_service.features.kernel.learned.features import (
    REQUIRED_INPUT_COLUMNS as REQUIRED_INPUT_COLUMNS,
)
from quant_platform.services.research_service.features.kernel.learned.features import (
    compute_learned_features as compute_learned_features,
)
