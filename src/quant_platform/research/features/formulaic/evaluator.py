"""Compatibility shim — formulaic evaluator moved to the inner-layer kernel (ADR-011).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.formulaic.evaluator``.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.formulaic.evaluator import (
    ExpressionCache as ExpressionCache,
)
from quant_platform.services.research_service.features.kernel.formulaic.evaluator import (
    evaluate_expression as evaluate_expression,
)

__all__ = [
    "ExpressionCache",
    "evaluate_expression",
]
