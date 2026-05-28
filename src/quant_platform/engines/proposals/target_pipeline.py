"""Shared signal-to-target pipeline for engine modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.engines.runtime.regime import detect_session_regime

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.application.runtime.state import Session
    from quant_platform.core.domain.portfolio import PortfolioTarget
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.research.runs import StrategyRun
    from quant_platform.core.domain.signals import RegimeState, SignalScore


@dataclass(frozen=True)
class EngineTargetBuild:
    """Result of one signal/regime/target build pass."""

    account: AccountSnapshot
    signals: list[SignalScore]
    regime: RegimeState
    target: PortfolioTarget | None


async def build_engine_target(
    *,
    session: Session,
    strategy_run: StrategyRun,
    feature_data: dict[uuid.UUID, dict[str, float]],
    as_of: datetime,
) -> EngineTargetBuild:
    """Sync account state, generate signals, refresh regime, and build target."""
    if (
        session.signal_ctrl is None
        or session.portfolio_ctrl is None
        or session.regime_detector is None
    ):
        raise RuntimeError("engine target session is missing required controllers")

    account = await session.account_broker.sync_account()
    signals = await session.signal_ctrl.generate(
        feature_data=feature_data,
        strategy_run=strategy_run,
        as_of=as_of,
    )
    regime = await detect_session_regime(session, as_of)
    target = await session.portfolio_ctrl.build(
        signals=signals,
        regime=regime,
        account=account,
        limits=session.risk_limits,
    )
    return EngineTargetBuild(
        account=account,
        signals=list(signals),
        regime=regime,
        target=target,
    )
