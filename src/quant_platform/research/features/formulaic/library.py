"""Compatibility shim — formulaic library moved to the inner-layer kernel (ADR-011).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.formulaic.library``.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.formulaic.library import (
    LIBRARY as LIBRARY,
)
from quant_platform.services.research_service.features.kernel.formulaic.library import (
    FormulaicAlpha as FormulaicAlpha,
)

__all__ = [
    "FormulaicAlpha",
    "LIBRARY",
]
