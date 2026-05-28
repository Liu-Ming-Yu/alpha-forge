"""Compatibility exports for operator read models.

DTOs/protocols live in ``read_model_types`` and the projection implementation
lives in ``read_model_builder``.  This module keeps the historical import path
stable for CLI, API, and tests.
"""

from __future__ import annotations

from quant_platform.application.operator_api.read_model_builder import OperatorReadModelBuilder
from quant_platform.application.operator_api.read_model_types import (
    BlotterEntry,
    BlotterView,
    BrokerHealthView,
    CashLedgerViewPort,
    CashStatusView,
    CombinedExposureView,
    EngineBudgetView,
    ForecastEvidenceView,
    OrderAllocationView,
    PaperGateMetricsView,
    RegimeStateView,
    SignalContributionView,
    SignalDecayView,
    StrategyHealth,
    StrategyLifecycleView,
    ThrottleStateViewPort,
)

__all__ = [
    "BlotterEntry",
    "BlotterView",
    "BrokerHealthView",
    "CashLedgerViewPort",
    "CashStatusView",
    "CombinedExposureView",
    "EngineBudgetView",
    "ForecastEvidenceView",
    "OperatorReadModelBuilder",
    "OrderAllocationView",
    "PaperGateMetricsView",
    "RegimeStateView",
    "SignalContributionView",
    "SignalDecayView",
    "StrategyHealth",
    "StrategyLifecycleView",
    "ThrottleStateViewPort",
]
