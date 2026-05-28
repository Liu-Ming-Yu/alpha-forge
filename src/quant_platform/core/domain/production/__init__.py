"""Compatibility exports for production-readiness domain models.

The bounded production domain types live in focused modules so readiness,
performance gates, promotion candidates, execution proposals, and operator
evidence can evolve independently.  This module keeps the historical import
surface stable for adapters, CLI commands, and tests.
"""

from __future__ import annotations

from quant_platform.core.domain.production.candidate import (
    ProductionCandidateReport,
    PromotionMode,
)
from quant_platform.core.domain.production.execution import (
    CombinedPortfolioTarget,
    EngineBudget,
    EngineTargetContribution,
    EngineTargetProposal,
    ExecutionTacticPolicy,
    OrderAllocation,
    SignalContribution,
)
from quant_platform.core.domain.production.operator import (
    AlertEvent,
    AlphaReadinessEvidence,
    OperatorAction,
    OperatorApiKey,
    RunbookEvidence,
)
from quant_platform.core.domain.production.performance import (
    InstrumentPnl,
    MetricRollupSnapshot,
    NavSnapshot,
    PerformanceReport,
    ShadowPaperParityRecord,
    ShadowPaperParityStatus,
    SignalGateRecord,
    SignalGateStatus,
    TextSignalGateRecord,
    TextSignalGateStatus,
)
from quant_platform.core.domain.production.prediction import (
    ForecastEvidence,
    PredictionResult,
)
from quant_platform.core.domain.production.readiness import (
    BrokerHealthObservation,
    BrokerSmokeObservation,
    DataHealthInstrumentStatus,
    DataHealthReport,
    PaperLifecycleObservation,
    PreflightCheck,
    PreflightReport,
    ProductionProfile,
    ProductionReadinessReport,
    ReadinessSnapshot,
    ReadinessState,
    RuntimeHeartbeat,
)

__all__ = [
    "AlertEvent",
    "AlphaReadinessEvidence",
    "BrokerHealthObservation",
    "BrokerSmokeObservation",
    "CombinedPortfolioTarget",
    "DataHealthInstrumentStatus",
    "DataHealthReport",
    "EngineBudget",
    "EngineTargetContribution",
    "EngineTargetProposal",
    "ExecutionTacticPolicy",
    "ForecastEvidence",
    "InstrumentPnl",
    "MetricRollupSnapshot",
    "NavSnapshot",
    "OperatorAction",
    "OperatorApiKey",
    "OrderAllocation",
    "PaperLifecycleObservation",
    "PerformanceReport",
    "PreflightCheck",
    "PredictionResult",
    "PreflightReport",
    "ProductionCandidateReport",
    "ProductionProfile",
    "ProductionReadinessReport",
    "PromotionMode",
    "ReadinessSnapshot",
    "ReadinessState",
    "RunbookEvidence",
    "ShadowPaperParityRecord",
    "ShadowPaperParityStatus",
    "SignalContribution",
    "SignalGateRecord",
    "SignalGateStatus",
    "TextSignalGateRecord",
    "TextSignalGateStatus",
    "RuntimeHeartbeat",
]
