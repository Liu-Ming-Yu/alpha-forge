"""Compatibility exports for portfolio domain models.

Portfolio value objects are split into target construction DTOs and risk /
optimizer DTOs.  This module remains the stable import surface for services,
adapters, and tests.
"""

from __future__ import annotations

from quant_platform.core.domain.portfolio.risk import (
    CapitalBudget,
    OptimizerResult,
    PortfolioRiskModel,
    RiskLimits,
    RiskSnapshot,
    StressScenario,
)
from quant_platform.core.domain.portfolio.targets import PortfolioTarget

__all__ = [
    "CapitalBudget",
    "OptimizerResult",
    "PortfolioRiskModel",
    "PortfolioTarget",
    "RiskLimits",
    "RiskSnapshot",
    "StressScenario",
]
