"""Compatibility shim — learned-PCA config moved to the inner-layer kernel (ADR-012).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.learned.config``.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.learned.config import (
    DEFAULT_CONFIG as DEFAULT_CONFIG,
)
from quant_platform.services.research_service.features.kernel.learned.config import (
    DEFAULT_N_COMPONENTS as DEFAULT_N_COMPONENTS,
)
from quant_platform.services.research_service.features.kernel.learned.config import (
    FEATURE_SET_VERSION as FEATURE_SET_VERSION,
)
from quant_platform.services.research_service.features.kernel.learned.config import (
    LearnedConfig as LearnedConfig,
)
