"""Strategy plugin registry.

Plugins are allowed to define alpha features, signal models, and portfolio
constructors.  They do not receive broker handles and cannot submit orders;
all execution still flows through the shared session risk and broker stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from quant_platform.engines.engine_runner import (
    EngineConfig,
    EngineRunner,
    ExecutionBackend,
    RunMode,
)
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)
from quant_platform.services.portfolio_service.vol_sizing import (
    VolTargetedPortfolioConstructor,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from decimal import Decimal

    from quant_platform.config import PlatformSettings


@dataclass(frozen=True)
class FeatureSpec:
    """Feature contract a plugin expects before it can score instruments."""

    name: str
    version: str
    required_features: tuple[str, ...]


class StrategyPlugin(Protocol):
    """Formal interface for production strategy plugins."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def feature_spec(self) -> FeatureSpec: ...

    def build_signal_model(
        self,
        factor_weights: Mapping[str, float],
        model_version: str,
    ) -> LinearWeightSignalModel: ...

    def build_portfolio_constructor(
        self,
        settings: PlatformSettings,
        max_positions: int,
    ) -> LongOnlyPortfolioConstructor: ...

    def create_runner(
        self,
        *,
        run_mode: RunMode,
        initial_cash: Decimal,
        settings: PlatformSettings | None,
        factor_weights: dict[str, float] | None,
        max_positions: int | None,
        rebalance_interval_seconds: float | None,
        instrument_contracts: dict[uuid.UUID, dict[str, object]] | None,
        execution_backend: ExecutionBackend,
    ) -> EngineRunner: ...


@dataclass(frozen=True)
class BuiltInStrategyPlugin:
    """Built-in long-only strategy plugin backed by the standard engine runner."""

    name: str
    version: str
    feature_spec: FeatureSpec
    default_factor_weights: Mapping[str, float]
    default_max_positions: int
    default_rebalance_interval_seconds: float
    default_universe_symbols: tuple[str, ...] = ()

    def build_signal_model(
        self,
        factor_weights: Mapping[str, float],
        model_version: str,
    ) -> LinearWeightSignalModel:
        return LinearWeightSignalModel(factor_weights, model_version=model_version)

    def build_portfolio_constructor(
        self,
        settings: PlatformSettings,
        max_positions: int,
    ) -> LongOnlyPortfolioConstructor:
        if settings.vol_sizing.enabled:
            return VolTargetedPortfolioConstructor(
                top_n=max_positions,
                vol_target=settings.vol_sizing.vol_target_annualized,
                min_vol_floor=settings.vol_sizing.min_vol_floor,
            )
        return LongOnlyPortfolioConstructor(top_n=max_positions)

    def create_runner(
        self,
        *,
        run_mode: RunMode,
        initial_cash: Decimal,
        settings: PlatformSettings | None,
        factor_weights: dict[str, float] | None,
        max_positions: int | None,
        rebalance_interval_seconds: float | None,
        instrument_contracts: dict[uuid.UUID, dict[str, object]] | None,
        execution_backend: ExecutionBackend,
    ) -> EngineRunner:
        weights = dict(factor_weights or self.default_factor_weights)
        config = EngineConfig(
            engine_name=self.name,
            engine_version=self.version,
            run_mode=run_mode,
            execution_backend=execution_backend,
            initial_cash=initial_cash,
            factor_weights=weights,
            max_positions=max_positions or self.default_max_positions,
            rebalance_interval_seconds=(
                rebalance_interval_seconds or self.default_rebalance_interval_seconds
            ),
            instrument_contracts=instrument_contracts or {},
            plugin_name=self.name,
            feature_set_name=self.feature_spec.name,
            required_features=self.feature_spec.required_features,
            signal_model_factory=lambda w, v: self.build_signal_model(w, v),
            portfolio_constructor_factory=self.build_portfolio_constructor,
        )
        return EngineRunner(config, settings)


_CROSS_SECTIONAL_FEATURES = (
    "momentum_1m",
    "momentum_3m",
    "momentum_12m_1m",
    "vol_compression",
)
_ETF_FEATURES = (
    "trend_quality_63d",
    "momentum_3m",
    "momentum_12m_1m",
    "vol_compression",
    "distance_to_52w_high",
)

_PLUGINS: dict[str, BuiltInStrategyPlugin] = {
    "cross_sectional_equity": BuiltInStrategyPlugin(
        name="cross_sectional_equity_v1",
        version="0.1.0",
        feature_spec=FeatureSpec(
            name="classical-cross-sectional",
            version="1.0.0",
            required_features=_CROSS_SECTIONAL_FEATURES,
        ),
        default_factor_weights={
            "momentum_1m": 0.20,
            "momentum_3m": 0.30,
            "momentum_12m_1m": 0.40,
            "vol_compression": 0.10,
        },
        default_max_positions=20,
        default_rebalance_interval_seconds=300.0,
    ),
    "etf_macro_allocator": BuiltInStrategyPlugin(
        name="etf_macro_allocator_v1",
        version="0.1.0",
        feature_spec=FeatureSpec(
            name="classical-etf-macro",
            version="1.0.0",
            required_features=_ETF_FEATURES,
        ),
        default_factor_weights={
            "trend_quality_63d": 0.35,
            "momentum_3m": 0.30,
            "momentum_12m_1m": 0.20,
            "vol_compression": 0.10,
            "distance_to_52w_high": 0.05,
        },
        default_max_positions=4,
        default_rebalance_interval_seconds=86400.0,
        default_universe_symbols=("SPY", "QQQ", "IWM", "TLT", "GLD", "XLK", "XLF", "XLE", "XLV"),
    ),
}


def list_strategy_plugins() -> tuple[str, ...]:
    """Return configured plugin keys usable from the CLI."""
    return tuple(sorted(_PLUGINS))


def get_strategy_plugin(name: str) -> StrategyPlugin:
    """Return a strategy plugin by CLI name."""
    try:
        return _PLUGINS[name]
    except KeyError as exc:
        valid = ", ".join(list_strategy_plugins())
        raise ValueError(f"unknown strategy plugin {name!r}; valid plugins: {valid}") from exc


def create_engine_from_plugin(
    name: str,
    *,
    run_mode: RunMode,
    initial_cash: Decimal,
    settings: PlatformSettings | None = None,
    factor_weights: dict[str, float] | None = None,
    max_positions: int | None = None,
    rebalance_interval_seconds: float | None = None,
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
    execution_backend: ExecutionBackend = ExecutionBackend.SIMULATED,
) -> EngineRunner:
    """Create an EngineRunner from the formal strategy plugin registry."""
    plugin = get_strategy_plugin(name)
    return plugin.create_runner(
        run_mode=run_mode,
        initial_cash=initial_cash,
        settings=settings,
        factor_weights=factor_weights,
        max_positions=max_positions,
        rebalance_interval_seconds=rebalance_interval_seconds,
        instrument_contracts=instrument_contracts,
        execution_backend=execution_backend,
    )
