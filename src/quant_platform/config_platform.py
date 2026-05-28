"""Root platform settings composition."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from quant_platform.config_api import ApiSettings
from quant_platform.config_broker import BrokerSettings
from quant_platform.config_data import DataIngestSettings
from quant_platform.config_governance_models import (
    BacktestSettings,
    ProductionSettings,
    RegimeSettings,
    V2Settings,
)
from quant_platform.config_logging import LoggingSettings
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


class PlatformSettings(BaseSettings):
    """Root configuration object for the quant platform.

    Reads from a ``.env`` file and/or OS environment variables prefixed
    with ``QP__``.  Nested models use double-underscore separators::

        QP__BROKER__PORT=7497
        QP__RISK__MAX_SINGLE_NAME_WEIGHT=0.10
        QP__THROTTLE__CAPACITY=20
    """

    model_config = SettingsConfigDict(
        env_prefix="QP__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    broker: BrokerSettings = BrokerSettings()
    risk: RiskSettings = RiskSettings()
    throttle: ThrottleSettings = ThrottleSettings()
    execution: ExecutionSettings = ExecutionSettings()
    cash: CashSettings = CashSettings()
    storage: StorageSettings = StorageSettings()
    logging: LoggingSettings = LoggingSettings()
    factors: FactorSettings = FactorSettings()
    vol_sizing: VolSizingSettings = VolSizingSettings()
    liquidity: LiquiditySettings = LiquiditySettings()
    api: ApiSettings = ApiSettings()
    regime: RegimeSettings = RegimeSettings()
    backtest: BacktestSettings = BacktestSettings()
    data_ingest: DataIngestSettings = DataIngestSettings()
    llm: LLMSettings = LLMSettings()
    boosting: BoostingSettings = BoostingSettings()
    alpha: AlphaSettings = AlphaSettings()
    production: ProductionSettings = ProductionSettings()
    v2: V2Settings = V2Settings()
    allow_dev_defaults: bool = Field(
        default=False,
        description=(
            "Escape hatch for ``_assert_live_session_defaults``.  When True, "
            "a live session may start with ``Simple*`` / ``InMemory*`` "
            "implementations (regime detector, order repo, event bus, ...) "
            "that would otherwise fail-closed at startup.  Never set this in "
            "production; it exists for CI smoke tests and ad-hoc debugging."
        ),
    )

    @model_validator(mode="after")
    def _live_requires_trading_hours_enforced(self) -> PlatformSettings:
        if not self.broker.paper_trading and not self.execution.trading_hours_enforced:
            raise ValueError(
                "trading_hours_enforced must be True when paper_trading=False. "
                "Set QP__EXECUTION__TRADING_HOURS_ENFORCED=true for live deployments."
            )
        return self

    @model_validator(mode="after")
    def _industrial_profile_requires_production_gates(self) -> PlatformSettings:
        if self.production.profile_preset != "industrial":
            return self
        missing: list[str] = []
        if not self.storage.postgres_dsn:
            missing.append("QP__STORAGE__POSTGRES_DSN")
        if not self.storage.redis_url:
            missing.append("QP__STORAGE__REDIS_URL")
        if self.storage.event_bus_backend != "redis_streams":
            missing.append("QP__STORAGE__EVENT_BUS_BACKEND=redis_streams")
        if not self.api.operator_api_key.strip():
            missing.append("QP__API__OPERATOR_API_KEY")
        if self.api.allow_unauthenticated or self.api.acknowledge_unauthenticated_risk:
            missing.append("authenticated operator API")
        if self.liquidity.allow_missing_profile:
            missing.append("QP__LIQUIDITY__ALLOW_MISSING_PROFILE=false")
        if not self.risk.require_sector_mapping:
            missing.append("QP__RISK__REQUIRE_SECTOR_MAPPING=true")
        if not self.risk.require_registered_model_match:
            missing.append("QP__RISK__REQUIRE_REGISTERED_MODEL_MATCH=true")
        if not self.execution.trading_hours_enforced:
            missing.append("QP__EXECUTION__TRADING_HOURS_ENFORCED=true")
        if not self.regime.require_seed_on_cycle:
            missing.append("QP__REGIME__REQUIRE_SEED_ON_CYCLE=true")
        if not self.regime.market_proxy_instrument_id:
            missing.append("QP__REGIME__MARKET_PROXY_INSTRUMENT_ID")
        if not self.v2.enabled:
            missing.append("QP__V2__ENABLED=true")
        if not self.v2.account_orchestrator_enabled:
            missing.append("QP__V2__ACCOUNT_ORCHESTRATOR_ENABLED=true")
        if not self.v2.require_dataset_quorum or not self.v2.third_eod_vendor.strip():
            missing.append("QP__V2__REQUIRE_DATASET_QUORUM=true with third vendor")
        if not self.v2.require_event_sourced_oms:
            missing.append("QP__V2__REQUIRE_EVENT_SOURCED_OMS=true")
        if not self.v2.readiness_snapshot_required:
            missing.append("QP__V2__READINESS_SNAPSHOT_REQUIRED=true")
        if missing:
            raise ValueError(
                "production.profile_preset='industrial' requires: " + ", ".join(missing)
            )
        return self
