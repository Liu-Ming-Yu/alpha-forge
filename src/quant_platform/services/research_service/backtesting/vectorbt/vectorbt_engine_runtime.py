"""Runtime dependency construction for the VectorBT backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..simple.backtest_execution_model import (
    BacktestExecutionModel,
)
from .vectorbt_portfolio_simulator import (
    VectorBTPortfolioSimulator,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import LiquidityProfileProvider, PortfolioConstructor

    from ..slippage import (
        IBKRCommissionSchedule,
        SlippageModel,
    )


@dataclass(frozen=True)
class VectorBTRuntime:
    """Runtime collaborators used by VectorBTBacktestEngine."""

    execution_model: BacktestExecutionModel
    portfolio_simulator: VectorBTPortfolioSimulator
    fallback_logged: set[uuid.UUID]


def build_vectorbt_runtime(
    *,
    settings: PlatformSettings,
    slippage_model: SlippageModel,
    commission_schedule: IBKRCommissionSchedule,
    universe_manager: LiquidityProfileProvider | None,
    portfolio_constructor: PortfolioConstructor,
    parity_mode: bool,
) -> VectorBTRuntime:
    fallback_logged: set[uuid.UUID] = set()
    execution_model = BacktestExecutionModel(
        settings=settings,
        slippage_model=slippage_model,
        commission_schedule=commission_schedule,
        universe_manager=universe_manager,
        fallback_logged=fallback_logged,
    )
    portfolio_simulator = VectorBTPortfolioSimulator(
        execution_model=execution_model,
        slippage_model=slippage_model,
        commission_schedule=commission_schedule,
        portfolio_constructor=portfolio_constructor,
        parity_mode=parity_mode,
    )
    return VectorBTRuntime(
        execution_model=execution_model,
        portfolio_simulator=portfolio_simulator,
        fallback_logged=fallback_logged,
    )
