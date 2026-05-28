"""Bootstrap factories for research backtest engine wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from quant_platform.bootstrap.session.public_api import create_paper_session
from quant_platform.engines.session.public_api import run_strategy_cycle

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        Clock,
        LiquidityProfileProvider,
        PaperSessionFactory,
        PortfolioConstructor,
        SignalModel,
        StrategyCycleRunner,
    )
    from quant_platform.services.research_service.backtesting.simple.backtest_engine import (
        SimpleBacktestEngine,
    )
    from quant_platform.services.research_service.backtesting.slippage import (
        IBKRCommissionSchedule,
        SlippageModel,
    )
    from quant_platform.services.research_service.backtesting.vectorbt.vectorbt_engine import (
        VectorBTBacktestEngine,
    )
    from quant_platform.services.research_service.intraday.backtesting.backtest import (
        IntradayBacktestEngine,
        VectorizedIntradayBacktestEngine,
    )
    from quant_platform.services.research_service.intraday.replay.replay import (
        IntradayTacticReplayModel,
    )


def create_simple_backtest_engine(
    *,
    clock: Clock,
    signal_model: SignalModel | None = None,
    portfolio_constructor: PortfolioConstructor | None = None,
    settings: PlatformSettings | None = None,
    slippage_model: SlippageModel | None = None,
    commission_schedule: IBKRCommissionSchedule | None = None,
    universe_manager: LiquidityProfileProvider | None = None,
) -> SimpleBacktestEngine:
    """Create a simple backtest engine with live-like runtime hooks injected."""
    from quant_platform.services.research_service.backtesting.simple.backtest_engine import (
        SimpleBacktestEngine,
    )

    return SimpleBacktestEngine(
        clock=clock,
        signal_model=signal_model,
        portfolio_constructor=portfolio_constructor,
        settings=settings,
        slippage_model=slippage_model,
        commission_schedule=commission_schedule,
        universe_manager=universe_manager,
        paper_session_factory=cast("PaperSessionFactory", create_paper_session),
        strategy_cycle_runner=cast("StrategyCycleRunner", run_strategy_cycle),
    )


def create_vectorbt_backtest_engine(
    *,
    clock: Clock,
    signal_model: SignalModel | None = None,
    portfolio_constructor: PortfolioConstructor | None = None,
    settings: PlatformSettings | None = None,
    slippage_model: SlippageModel | None = None,
    commission_schedule: IBKRCommissionSchedule | None = None,
    universe_manager: LiquidityProfileProvider | None = None,
    top_n: int = 10,
    parity_mode: bool = False,
) -> VectorBTBacktestEngine:
    """Create the daily VectorBT engine through the bootstrap composition root."""
    from quant_platform.services.research_service.backtesting.vectorbt.vectorbt_engine import (
        VectorBTBacktestEngine,
    )

    return VectorBTBacktestEngine(
        clock=clock,
        signal_model=signal_model,
        portfolio_constructor=portfolio_constructor,
        settings=settings,
        slippage_model=slippage_model,
        commission_schedule=commission_schedule,
        universe_manager=universe_manager,
        top_n=top_n,
        parity_mode=parity_mode,
    )


def create_intraday_backtest_engine(
    *,
    settings: PlatformSettings | None = None,
    replay_model: IntradayTacticReplayModel | None = None,
    portfolio_constructor: PortfolioConstructor | None = None,
    signal_model: object | None = None,
) -> IntradayBacktestEngine:
    """Create the canonical intraday engine with session construction injected."""
    from quant_platform.services.research_service.intraday.backtesting.backtest import (
        IntradayBacktestEngine,
    )

    return IntradayBacktestEngine(
        settings=settings,
        replay_model=replay_model,
        portfolio_constructor=portfolio_constructor,
        signal_model=signal_model,
        paper_session_factory=cast("PaperSessionFactory", create_paper_session),
    )


def create_vectorized_intraday_backtest_engine(
    *,
    settings: PlatformSettings | None = None,
    portfolio_constructor: PortfolioConstructor | None = None,
    signal_model: object | None = None,
    commission_schedule: IBKRCommissionSchedule | None = None,
) -> VectorizedIntradayBacktestEngine:
    """Create the vectorized intraday comparator with session construction injected."""
    from quant_platform.services.research_service.intraday.backtesting.backtest import (
        VectorizedIntradayBacktestEngine,
    )

    return VectorizedIntradayBacktestEngine(
        settings=settings,
        portfolio_constructor=portfolio_constructor,
        signal_model=signal_model,
        commission_schedule=commission_schedule,
        paper_session_factory=cast("PaperSessionFactory", create_paper_session),
    )


__all__ = [
    "create_intraday_backtest_engine",
    "create_simple_backtest_engine",
    "create_vectorbt_backtest_engine",
    "create_vectorized_intraday_backtest_engine",
]
