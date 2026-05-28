"""Service contracts — every cross-boundary dependency in the platform.

All contracts are defined as typing.Protocol classes so concrete adapters
are decoupled from the interface definition.  No concrete types, ORMs, or
broker SDKs are imported in any contract module.

Design rules enforced by these contracts:
- No order may be emitted without a successful CashConstraintEngine check
  and a CashReservation.
- Broker state is authoritative; BrokerGateway.sync_* methods are the
  only source of truth for live account state.
- All outbound broker requests pass through ExecutionPolicy (throttling,
  kill switch) before reaching BrokerGateway.
- Live and simulated (research) paths share PortfolioConstructor, RiskPolicy,
  CashConstraintEngine, ExecutionPolicy, and OrderRepository unchanged.
  Only BrokerGateway is swapped.
- EventBus delivery is at-least-once; all consumers must be idempotent.

Adding a new contract:
  1. Define it in the bounded-context module that owns it (data, research,
     portfolio, execution, repositories, infrastructure) or in ``common`` if
     it is a value type shared across contexts.
  2. Document invariants and what the contract must *never* do.
  3. Re-export the symbol from this package so call sites keep using
     ``from quant_platform.core.contracts import ...``.
  4. Write the concrete adapter in the relevant service package.
  5. Wire at startup via dependency injection — never instantiate adapters
     inside controllers or domain services.

This package is a pure re-export of the bounded-context submodules; it
contains no behavior.  The previous monolithic ``contracts.py`` has been
split to reduce the single-file blast radius.
"""

from quant_platform.core.contracts.common import (
    BrokerAck,
    BrokerCapabilities,
    BrokerHealth,
    BrokerHealthStatus,
    TradeDecision,
)
from quant_platform.core.contracts.data import (
    DatasetCatalog,
    FeatureRepository,
    HistoricalBarVendorAdapter,
    HistoricalDataStore,
    InstrumentRepository,
    LiquidityProfileProvider,
    LiquidityProfileSnapshot,
    MarketDataProvider,
    SecurityMaster,
)
from quant_platform.core.contracts.execution import (
    BrokerGateway,
    BrokerOrderRoutingGateway,
    BrokerSessionGateway,
    ExecutionPolicy,
    ExecutionRouter,
    LifecycleFeed,
    OrderStateStore,
)
from quant_platform.core.contracts.features import (
    FeatureComputer,
    FeatureFamilyDescriptor,
    FeatureFamilyPlugin,
    NamedPlugin,
)
from quant_platform.core.contracts.infrastructure import (
    ArtifactStore,
    AuditSink,
    Clock,
    EventBus,
)
from quant_platform.core.contracts.model_registry import (
    ModelRegistryRepository,
    RegisteredModelRecord,
)
from quant_platform.core.contracts.portfolio import (
    CashConstraintEngine,
    Optimizer,
    PortfolioConstructor,
    PortfolioRiskModelRepository,
    RegimeDetector,
    RegimeScaleProvider,
    RiskPolicy,
)
from quant_platform.core.contracts.production import (
    GovernanceEvidenceRepository,
    MultiEngineGovernanceRepository,
    NavSnapshotRepository,
    OperationalReadinessRepository,
    OperatorActionRepository,
    OperatorAuthRepository,
    PerformanceRepository,
    PredictionEvidenceRepository,
    ProductionEvidenceRepository,
    ShadowPaperParityRepository,
    SignalContributionRepository,
    SignalPromotionGate,
    TextSignalPromotionGate,
)
from quant_platform.core.contracts.redis import AsyncRedisClient
from quant_platform.core.contracts.repositories import (
    OrderRepository,
    PositionRepository,
)
from quant_platform.core.contracts.research import (
    AlphaSource,
    BacktestCycleResult,
    BacktestEngine,
    BacktestExecutionPlan,
    BacktestOrderIntentRepository,
    BacktestReplayBroker,
    BacktestSession,
    FeatureAuditRepository,
    ModelArtifactRepository,
    PaperSessionFactory,
    PromotionGate,
    SignalModel,
    StrategyCycleRunner,
)
from quant_platform.core.contracts.text_data import TextEventProvider

__all__ = [
    # common value types
    "BrokerAck",
    "BrokerCapabilities",
    "BrokerHealth",
    "BrokerHealthStatus",
    "TradeDecision",
    # data
    "DatasetCatalog",
    "FeatureRepository",
    "FeatureComputer",
    "FeatureFamilyDescriptor",
    "FeatureFamilyPlugin",
    "HistoricalBarVendorAdapter",
    "HistoricalDataStore",
    "InstrumentRepository",
    "LiquidityProfileProvider",
    "LiquidityProfileSnapshot",
    "MarketDataProvider",
    "NamedPlugin",
    "SecurityMaster",
    # research
    "AlphaSource",
    "BacktestCycleResult",
    "BacktestEngine",
    "BacktestExecutionPlan",
    "BacktestOrderIntentRepository",
    "BacktestReplayBroker",
    "BacktestSession",
    "FeatureAuditRepository",
    "ModelArtifactRepository",
    "PaperSessionFactory",
    "PromotionGate",
    "SignalModel",
    "StrategyCycleRunner",
    # portfolio / risk
    "CashConstraintEngine",
    "Optimizer",
    "PortfolioConstructor",
    "PortfolioRiskModelRepository",
    "RegimeDetector",
    "RegimeScaleProvider",
    "RiskPolicy",
    # production readiness
    "GovernanceEvidenceRepository",
    "ModelRegistryRepository",
    "NavSnapshotRepository",
    "OperatorAuthRepository",
    "OperatorActionRepository",
    "PerformanceRepository",
    "PredictionEvidenceRepository",
    "ShadowPaperParityRepository",
    "ProductionEvidenceRepository",
    "RegisteredModelRecord",
    "SignalContributionRepository",
    "SignalPromotionGate",
    "OperationalReadinessRepository",
    "TextSignalPromotionGate",
    "MultiEngineGovernanceRepository",
    # execution
    "BrokerGateway",
    "BrokerOrderRoutingGateway",
    "BrokerSessionGateway",
    "ExecutionRouter",
    "ExecutionPolicy",
    "LifecycleFeed",
    "OrderStateStore",
    # repositories
    "OrderRepository",
    "PositionRepository",
    # redis
    "AsyncRedisClient",
    # infrastructure
    "AuditSink",
    "ArtifactStore",
    "Clock",
    "EventBus",
    # text data (Phase 5)
    "TextEventProvider",
]
