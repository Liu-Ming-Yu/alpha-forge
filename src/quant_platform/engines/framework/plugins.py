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
from quant_platform.services.research_service.features.pv_formulaic.family import (
    PV_FORMULAIC_FEATURE_SET_VERSION,
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
    #: Feature-set version whose family the engine schedules + computes each cycle.
    #: Empty ⇒ the engine default (``close``); set it to bind a non-close family
    #: (e.g. ``pv-formulaic-live-v1`` for Arm G). See ``EngineConfig``.
    feature_set_version: str = ""

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
            feature_set_version=self.feature_set_version,
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

# Arm G (long_only_top30_pv_formulaic_streakdial) — the production research lead.
# Frozen IC-weighted-non-negative weights from the promoted evidence
# (``backtest_latest_stack_realized_v2/arm_long_only_top30_pv_formulaic_streakdial.json``,
# model_version ``ic-weighted-non-negative``). The 20 keys are the live
# pv_formulaic feature names (see ``PV_FORMULAIC_FEATURE_NAMES``); weights sum to 1.0.
_ARM_G_FACTOR_WEIGHTS: dict[str, float] = {
    "close_to_open_return": 0.023050530701360726,
    "distance_to_52w_high": 0.0061608259803869105,
    "dollar_volume_20d": 0.05718013891269491,
    "drawdown_from_252d_high": 0.0061608259803869105,
    "high_low_range_1d": 0.12232121378085248,
    "high_low_range_20d": 0.10903636583858742,
    "mom_12_1": 0.060791507163893914,
    "mom_3_1": 0.04851448819474293,
    "mom_6_1": 0.05551777223894792,
    "overnight_gap": 0.023050530701360726,
    "ret_126d": 0.05261728109533933,
    "ret_21d": 0.05025937382017044,
    "ret_252d": 0.06836019406551445,
    "ret_63d": 0.07751913862540603,
    "reversal_1d": 0.05964022188804077,
    "reversal_5d": 0.05224052713604494,
    "volume_z_20d": 0.08491219912439091,
    "wq_alpha_002_paraphrase": 0.0020047688478355375,
    "wq_alpha_012": 0.003010315784891496,
    "wq_alpha_041": 0.03765178011915117,
}
_ARM_G_FEATURES = tuple(sorted(_ARM_G_FACTOR_WEIGHTS))

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
    "arm_g": BuiltInStrategyPlugin(
        name="arm_g_pv_formulaic_v1",
        version="0.1.0",
        feature_spec=FeatureSpec(
            name="pv-formulaic-live",
            version=PV_FORMULAIC_FEATURE_SET_VERSION,
            required_features=_ARM_G_FEATURES,
        ),
        default_factor_weights=_ARM_G_FACTOR_WEIGHTS,
        default_max_positions=30,
        # Monthly cadence (~21 trading days) to track the backtest holding period
        # and its 0.48% turnover; increment 4 reconciles live turnover vs evidence.
        default_rebalance_interval_seconds=21 * 86400.0,
        # Bind the live pv_formulaic family so the engine computes G's 20 features
        # (not the default ``close`` family).
        feature_set_version=PV_FORMULAIC_FEATURE_SET_VERSION,
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
