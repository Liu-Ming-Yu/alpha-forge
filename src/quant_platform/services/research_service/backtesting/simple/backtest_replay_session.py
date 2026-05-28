"""Replay-session construction helpers for simple backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from quant_platform.services.research_service.backtesting.simple.replay_clock import FakeClock

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        BacktestReplayBroker,
        BacktestSession,
        PaperSessionFactory,
        PortfolioConstructor,
        RegimeDetector,
        SignalModel,
    )
    from quant_platform.core.domain.research import StrategyRun


@dataclass(frozen=True)
class BacktestReplaySession:
    """Runtime objects needed by the replay loop."""

    fake_clock: FakeClock
    session: BacktestSession | Any
    broker: BacktestReplayBroker
    history_closes: dict[uuid.UUID, list[float]]


def build_backtest_replay_settings(settings: PlatformSettings) -> PlatformSettings:
    """Return settings adjusted for historical replay."""
    return settings.model_copy(
        update={
            "execution": settings.execution.model_copy(update={"trading_hours_enforced": False}),
        }
    )


def create_backtest_replay_session(
    *,
    session_factory: PaperSessionFactory,
    settings: PlatformSettings,
    start: datetime,
    initial_capital: Decimal,
    strategy_run: StrategyRun,
    signal_model: SignalModel | None,
    portfolio_constructor: PortfolioConstructor | None,
    regime_detector: RegimeDetector,
) -> BacktestReplaySession:
    """Create the paper-session shell used to replay historical cycles."""
    fake_clock = FakeClock(initial=start)
    session = cast(
        "Any",
        session_factory(
            settings=build_backtest_replay_settings(settings),
            initial_cash=initial_capital,
            strategy_run_id=strategy_run.run_id,
            clock=fake_clock,
            signal_model=signal_model,
            portfolio_constructor=portfolio_constructor,
            regime_detector=regime_detector,
        ),
    )
    broker: BacktestReplayBroker = session.broker
    return BacktestReplaySession(
        fake_clock=fake_clock,
        session=session,
        broker=broker,
        history_closes={},
    )
