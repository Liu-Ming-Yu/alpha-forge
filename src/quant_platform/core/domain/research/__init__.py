"""Compatibility exports for research domain models.

Research value objects are split by bounded concern: run lifecycle,
feature governance, backtest evidence, and model readiness.  This module
keeps the historical import path stable while the service layer moves to
more explicit dependencies.
"""

from __future__ import annotations

from quant_platform.core.domain.research.backtests import (
    BacktestEvidenceManifest,
    BacktestReconciliationReport,
    BacktestReconciliationStatus,
    BacktestRun,
    IntradayBacktestSpec,
)
from quant_platform.core.domain.research.features import (
    FeatureAuditResult,
    FeatureDataset,
    FeatureDefinition,
    FeatureExpectedSign,
    FeatureProductionState,
    FeatureRequest,
    FeatureResult,
    FeatureSnapshot,
    FeatureVector,
)
from quant_platform.core.domain.research.models import (
    AlphaReadinessReport,
    ModelArtifact,
    ModelCard,
    PromotionState,
)
from quant_platform.core.domain.research.runs import (
    RunStatus,
    RunType,
    StrategyRun,
)
from quant_platform.core.domain.signals.feature_inputs import FeatureInputContext

__all__ = [
    "AlphaReadinessReport",
    "BacktestEvidenceManifest",
    "BacktestReconciliationReport",
    "BacktestReconciliationStatus",
    "BacktestRun",
    "FeatureAuditResult",
    "FeatureDataset",
    "FeatureDefinition",
    "FeatureExpectedSign",
    "FeatureInputContext",
    "FeatureProductionState",
    "FeatureRequest",
    "FeatureResult",
    "FeatureSnapshot",
    "FeatureVector",
    "IntradayBacktestSpec",
    "ModelArtifact",
    "ModelCard",
    "PromotionState",
    "RunStatus",
    "RunType",
    "StrategyRun",
]
