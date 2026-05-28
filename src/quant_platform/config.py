"""Compatibility facade for quant-platform settings.

Runtime settings are split into focused modules so adapters and bootstrap code
can depend on smaller ownership areas. Importing from ``quant_platform.config``
remains supported for existing CLI, API, and session entrypoints.
"""

from __future__ import annotations

from quant_platform.config_api import ApiSettings
from quant_platform.config_broker import BrokerSettings
from quant_platform.config_data import DataIngestSettings
from quant_platform.config_governance_models import (
    BacktestSettings,
    ProductionSettings,
    RegimeSettings,
    RegimeThresholdsSettings,
    V2Settings,
)
from quant_platform.config_logging import (
    LoggingSettings,
    configure_logging,
)
from quant_platform.config_platform import PlatformSettings
from quant_platform.config_risk_execution import (
    CashSettings,
    ExecutionSettings,
    LiquiditySettings,
    RiskSettings,
    ThrottleSettings,
)
from quant_platform.config_signal_models import (
    AlphaSettings,
    BoostingSettings,
    FactorSettings,
    LLMSettings,
    VolSizingSettings,
)
from quant_platform.config_storage import StorageSettings

__all__ = [
    "AlphaSettings",
    "ApiSettings",
    "BacktestSettings",
    "BoostingSettings",
    "BrokerSettings",
    "CashSettings",
    "DataIngestSettings",
    "ExecutionSettings",
    "FactorSettings",
    "LLMSettings",
    "LiquiditySettings",
    "LoggingSettings",
    "PlatformSettings",
    "ProductionSettings",
    "RegimeSettings",
    "RegimeThresholdsSettings",
    "RiskSettings",
    "StorageSettings",
    "ThrottleSettings",
    "V2Settings",
    "VolSizingSettings",
    "configure_logging",
]
