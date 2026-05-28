"""Initialization wiring helpers for engine runners."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.engines.framework.types import EngineConfig, RunMode
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)
from quant_platform.services.portfolio_service.vol_sizing import (
    VolTargetedPortfolioConstructor,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import Clock, SignalModel


def run_type_for_mode(run_mode: RunMode) -> RunType:
    """Map engine run mode to research run type."""
    return {
        RunMode.SHADOW: RunType.PAPER,
        RunMode.PAPER: RunType.PAPER,
        RunMode.LIVE: RunType.LIVE,
    }[run_mode]


def build_strategy_run(*, config: EngineConfig, clock: Clock) -> StrategyRun:
    """Build the StrategyRun record for an engine session."""
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name=config.engine_name,
        strategy_version=config.engine_version,
        run_type=run_type_for_mode(config.run_mode),
        status=RunStatus.RUNNING,
        config_snapshot={
            "factor_weights": config.factor_weights,
            "max_positions": config.max_positions,
            "run_mode": config.run_mode.value,
            "execution_backend": config.execution_backend.value,
        },
        created_at=clock.now(),
        started_at=clock.now(),
    )


def build_signal_model(config: EngineConfig) -> SignalModel:
    """Build the engine's signal model from config."""
    if config.signal_model_factory is not None:
        return config.signal_model_factory(
            config.factor_weights,
            config.engine_version,
        )
    return LinearWeightSignalModel(
        config.factor_weights,
        model_version=config.engine_version,
    )


def build_engine_portfolio_constructor(
    *,
    config: EngineConfig,
    settings: PlatformSettings,
) -> LongOnlyPortfolioConstructor:
    """Build the engine-scoped portfolio constructor."""
    if config.max_positions < 1:
        raise ValueError("EngineConfig.max_positions must be >= 1")
    if config.portfolio_constructor_factory is not None:
        return config.portfolio_constructor_factory(
            settings,
            config.max_positions,
        )
    if settings.vol_sizing.enabled:
        return VolTargetedPortfolioConstructor(
            top_n=config.max_positions,
            vol_target=settings.vol_sizing.vol_target_annualized,
            min_vol_floor=settings.vol_sizing.min_vol_floor,
        )
    return LongOnlyPortfolioConstructor(top_n=config.max_positions)
