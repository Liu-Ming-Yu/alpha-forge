"""Compatibility shim — formulaic config moved to the inner-layer kernel (ADR-011).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.formulaic.config``.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.formulaic.config import (
    DEFAULT_CONFIG as DEFAULT_CONFIG,
)
from quant_platform.services.research_service.features.kernel.formulaic.config import (
    FEATURE_SET_VERSION as FEATURE_SET_VERSION,
)
from quant_platform.services.research_service.features.kernel.formulaic.config import (
    OPERATOR_SET_VERSION as OPERATOR_SET_VERSION,
)
from quant_platform.services.research_service.features.kernel.formulaic.config import (
    FormulaicConfig as FormulaicConfig,
)

__all__ = [
    "DEFAULT_CONFIG",
    "FEATURE_SET_VERSION",
    "FormulaicConfig",
    "OPERATOR_SET_VERSION",
]
