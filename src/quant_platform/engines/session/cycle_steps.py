"""Composable steps for one strategy cycle."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, cast

import structlog

from quant_platform.core.domain.orders.lifecycle import BrokerFillEvent
from quant_platform.core.events import RegimeStateDetected
from quant_platform.engines.session.cycle_guards import empty_cycle_result
from quant_platform.engines.session.passive_reprice import run_passive_reprice_once
from quant_platform.services.portfolio_service.vol_sizing import (
    VolTargetedPortfolioConstructor,
)
from quant_platform.services.research_service.features.pipeline.feature_pipeline import (
    extract_vol_forecasts,
)
from quant_platform.services.signal_service.regime_detector import MarketRegimeDetector
from quant_platform.telemetry.metrics import time_phase

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal
    from typing import Protocol

    from quant_platform.application.runtime.state import CycleResult, Session
    from quant_platform.core.domain.orders import FillEvent
    from quant_platform.core.domain.orders.intent import OrderIntent
    from quant_platform.core.domain.portfolio import PortfolioTarget
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.research import StrategyRun
    from quant_platform.core.domain.signals import RegimeState, SignalScore
    from quant_platform.engines.session.strategy_cycle_types import MarketStatsReader
    from quant_platform.services.portfolio_service.portfolio_constructor import SimpleRegimeDetector

    class RegimeStateDetector(Protocol):
        async def detect(self, as_of: datetime) -> RegimeState: ...


log = structlog.get_logger(__name__)


async def apply_cycle_passive_reprice(
    *,
    session: Session,
    market_prices: dict[uuid.UUID, Decimal],
    engine_name: str,
) -> None:
    """Cancel due stale passive limits before planning fresh orders."""
    with time_phase(engine_name, "passive_reprice"):
        decisions = await run_passive_reprice_once(
            session=session,
            market_prices=market_prices,
        )
        if decisions:
            await drain_cycle_lifecycle_events(
                session=session,
                engine_name=engine_name,
            )


async def generate_cycle_signals(
    *,
    session: Session,
    feature_data: dict[uuid.UUID, dict[str, float]],
    strategy_run: StrategyRun,
    as_of: datetime,
    engine_name: str,
) -> list[SignalScore]:
    with time_phase(engine_name, "signals"):
        signal_ctrl = session.signal_ctrl
        if signal_ctrl is None:
            raise RuntimeError("session signal controller is not initialized")
        return cast(
            "list[SignalScore]",
            await signal_ctrl.generate(
                feature_data=feature_data,
                strategy_run=strategy_run,
                as_of=as_of,
            ),
        )


async def detect_cycle_regime(
    *,
    session: Session,
    supplied_regime: RegimeState | None,
    as_of: datetime,
    engine_name: str,
    market_stats_reader: MarketStatsReader,
) -> tuple[RegimeState | None, CycleResult | None]:
    with time_phase(engine_name, "regime"):
        regime_detector = session.regime_detector
        if supplied_regime is None and regime_detector is None:
            raise RuntimeError("session regime detector is not initialized")
        if supplied_regime is None and isinstance(regime_detector, MarketRegimeDetector):
            stats = await market_stats_reader(session, as_of)
            if stats is not None:
                regime_detector.update(stats)
            elif session.settings.regime.require_seed_on_cycle:
                log.error(
                    "strategy_cycle.regime_seed_missing",
                    proxy_instrument_id=session.settings.regime.market_proxy_instrument_id,
                )
                return None, empty_cycle_result()
        active_regime = (
            supplied_regime
            if supplied_regime is not None
            else await _detect_required_regime(_require_regime_detector(regime_detector), as_of)
        )
    return active_regime, None


def _require_regime_detector(
    regime_detector: SimpleRegimeDetector | MarketRegimeDetector | None,
) -> RegimeStateDetector:
    if regime_detector is None:
        raise RuntimeError("session regime detector is not initialized")
    return regime_detector


async def _detect_required_regime(
    regime_detector: RegimeStateDetector,
    as_of: datetime,
) -> RegimeState:
    return await regime_detector.detect(as_of)


async def publish_cycle_regime(session: Session, regime: RegimeState) -> None:
    gross_scale_map = {
        "risk_on": 1.0,
        "risk_off": 0.5,
        "transition": 0.75,
        "crisis": 0.0,
        "unknown": 1.0,
    }
    regime_label_str = regime.regime_label.value
    await session.event_bus.publish(
        RegimeStateDetected(
            event_id=uuid.uuid4(),
            occurred_at=regime.as_of,
            regime_id=regime.regime_id,
            regime_label=regime.regime_label,
            confidence=regime.confidence,
            gross_exposure_scale=gross_scale_map.get(regime_label_str, 1.0),
            supporting_features=dict(regime.supporting_features),
        )
    )


def refresh_cycle_vol_forecasts(
    session: Session,
    feature_data: dict[uuid.UUID, dict[str, float]],
) -> None:
    if isinstance(session.portfolio_constructor, VolTargetedPortfolioConstructor):
        forecasts = extract_vol_forecasts(feature_data)
        session.portfolio_constructor.set_vol_forecasts(forecasts)


async def build_cycle_portfolio_target(
    *,
    session: Session,
    signals: list[SignalScore],
    regime: RegimeState,
    account: AccountSnapshot,
    engine_name: str,
) -> PortfolioTarget | None:
    with time_phase(engine_name, "portfolio"):
        portfolio_ctrl = session.portfolio_ctrl
        if portfolio_ctrl is None:
            raise RuntimeError("session portfolio controller is not initialized")
        return cast(
            "PortfolioTarget | None",
            await portfolio_ctrl.build(
                signals=signals,
                regime=regime,
                account=account,
                limits=session.risk_limits,
            ),
        )


def plan_cycle_orders(
    *,
    session: Session,
    target: PortfolioTarget,
    account: AccountSnapshot,
    market_prices: dict[uuid.UUID, Decimal],
    strategy_run: StrategyRun,
    engine_name: str,
) -> list[OrderIntent]:
    with time_phase(engine_name, "planner"):
        order_planner = session.order_planner
        if order_planner is None:
            raise RuntimeError("session order planner is not initialized")
        return cast(
            "list[OrderIntent]",
            order_planner.plan(
                target=target,
                account=account,
                market_prices=market_prices,
                strategy_run_id=strategy_run.run_id,
            ),
        )


async def approve_cycle_orders(
    *,
    session: Session,
    intents: list[OrderIntent],
    account: AccountSnapshot,
    engine_name: str,
) -> tuple[list[OrderIntent], list[OrderIntent]]:
    with time_phase(engine_name, "gate"):
        return cast(
            "tuple[list[OrderIntent], list[OrderIntent]]",
            await session.approve_ctrl.approve(intents, account),
        )


async def submit_cycle_orders(
    *,
    session: Session,
    approved: list[OrderIntent],
    account: AccountSnapshot,
    engine_name: str,
) -> list[uuid.UUID]:
    with time_phase(engine_name, "submit"):
        return cast(
            "list[uuid.UUID]",
            await session.submit_ctrl.submit(approved, account=account),
        )


async def drain_cycle_lifecycle_events(
    *,
    session: Session,
    engine_name: str,
    wait_timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.1,
) -> list[FillEvent]:
    with time_phase(engine_name, "drain"):
        if session.lifecycle_feed is not None:
            return await _drain_lifecycle_feed(
                session=session,
                wait_timeout_seconds=wait_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )

    return []


async def _drain_lifecycle_feed(
    *,
    session: Session,
    wait_timeout_seconds: float,
    poll_interval_seconds: float,
) -> list[FillEvent]:
    """Drain lifecycle events, optionally polling briefly for late IB fills."""
    fills: list[FillEvent] = []
    timeout = max(0.0, wait_timeout_seconds)
    poll_interval = max(0.01, poll_interval_seconds)
    deadline = time.monotonic() + timeout

    while True:
        lifecycle_events = await session.lifecycle_feed.drain_lifecycle_events()
        if lifecycle_events:
            await session.coordinator.process_lifecycle_events(lifecycle_events)
            fills.extend(
                event.fill for event in lifecycle_events if isinstance(event, BrokerFillEvent)
            )

        if timeout <= 0 or time.monotonic() >= deadline:
            return fills

        await asyncio.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
