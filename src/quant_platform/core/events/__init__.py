"""Compatibility exports for domain events.

Domain events are split by service family, but this module remains the stable
import surface for event-bus serialization, adapters, CLI commands, and tests.
"""

from __future__ import annotations

from quant_platform.core.events.base import DomainEvent
from quant_platform.core.events.data import (
    CorporateActionRecorded,
    MarketBarIngested,
    TextEventIngested,
)
from quant_platform.core.events.execution import (
    BrokerSessionHealthChanged,
    CashDriftDetected,
    ComplianceViolationWarning,
    KillSwitchActivated,
    OrderCancelled,
    OrderFilled,
    OrderSubmissionUncertain,
    OrderSubmitted,
    OrphanOrderDetected,
    ReconciliationCompleted,
    SettlementApplied,
    UnmatchedFillEvent,
    UnsettledDebitAlert,
)
from quant_platform.core.events.portfolio import (
    OrderApproved,
    OrderRejected,
    PortfolioTargetBuilt,
)
from quant_platform.core.events.research import (
    BacktestCompleted,
    FeatureVectorComputed,
    RegimeStateDetected,
    SignalScorePublished,
)
from quant_platform.core.events.v2 import (
    AlertFired,
    AlertResolved,
    ApiKeyRevoked,
    EngineProposalGenerated,
    OperatorActionRecorded,
    OrderRouted,
    ProposalsMerged,
)

__all__ = [
    "AlertFired",
    "AlertResolved",
    "ApiKeyRevoked",
    "BacktestCompleted",
    "BrokerSessionHealthChanged",
    "CashDriftDetected",
    "ComplianceViolationWarning",
    "CorporateActionRecorded",
    "DomainEvent",
    "EngineProposalGenerated",
    "FeatureVectorComputed",
    "KillSwitchActivated",
    "MarketBarIngested",
    "OperatorActionRecorded",
    "OrderApproved",
    "OrderCancelled",
    "OrderFilled",
    "OrderRejected",
    "OrderRouted",
    "OrderSubmissionUncertain",
    "OrderSubmitted",
    "OrphanOrderDetected",
    "PortfolioTargetBuilt",
    "ProposalsMerged",
    "ReconciliationCompleted",
    "RegimeStateDetected",
    "SettlementApplied",
    "SignalScorePublished",
    "TextEventIngested",
    "UnmatchedFillEvent",
    "UnsettledDebitAlert",
]
