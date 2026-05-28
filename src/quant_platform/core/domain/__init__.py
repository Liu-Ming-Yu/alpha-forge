# quant_platform.core.domain — immutable domain entity definitions.
#
# All types here are frozen dataclasses (value objects).  They carry no
# behaviour beyond basic validation.  No SQLAlchemy, no Pydantic, no
# broker-specific fields.  Adapters are responsible for mapping to/from
# these types at the edges of the system.

from quant_platform.core.domain.instruments import CorporateAction, Instrument
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.market_data.text_events import TextEvent, TextEventType
from quant_platform.core.domain.orders import BrokerOrder, FillEvent, OrderIntent
from quant_platform.core.domain.portfolio import PortfolioTarget, RiskLimits
from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.core.domain.production import (
    CombinedPortfolioTarget,
    DataHealthInstrumentStatus,
    DataHealthReport,
    EngineBudget,
    EngineTargetContribution,
    ExecutionTacticPolicy,
    ForecastEvidence,
    MetricRollupSnapshot,
    NavSnapshot,
    OrderAllocation,
    PerformanceReport,
    PredictionResult,
    PreflightCheck,
    PreflightReport,
    ProductionProfile,
    TextSignalGateRecord,
    TextSignalGateStatus,
)
from quant_platform.core.domain.research import BacktestRun, FeatureVector, StrategyRun
from quant_platform.core.domain.settlement import CashReservation, SettlementLot
from quant_platform.core.domain.signals import RegimeState, SignalScore

__all__ = [
    "Instrument",
    "CorporateAction",
    "MarketBar",
    "FeatureVector",
    "SignalScore",
    "RegimeState",
    "PortfolioTarget",
    "RiskLimits",
    "OrderIntent",
    "BrokerOrder",
    "FillEvent",
    "PositionSnapshot",
    "AccountSnapshot",
    "SettlementLot",
    "CashReservation",
    "StrategyRun",
    "BacktestRun",
    "TextEvent",
    "TextEventType",
    "DataHealthInstrumentStatus",
    "DataHealthReport",
    "EngineBudget",
    "EngineTargetContribution",
    "CombinedPortfolioTarget",
    "OrderAllocation",
    "ExecutionTacticPolicy",
    "PredictionResult",
    "ForecastEvidence",
    "MetricRollupSnapshot",
    "NavSnapshot",
    "PerformanceReport",
    "PreflightCheck",
    "PreflightReport",
    "ProductionProfile",
    "TextSignalGateRecord",
    "TextSignalGateStatus",
]
