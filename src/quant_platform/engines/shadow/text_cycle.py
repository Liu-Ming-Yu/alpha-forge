"""Mode-specific helper for shadow text scoring."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable, Mapping
    from decimal import Decimal

    from quant_platform.application.runtime.state import Session
    from quant_platform.core.domain.market_data.text_events import TextEvent
    from quant_platform.core.domain.research.runs import StrategyRun
    from quant_platform.core.domain.signals import SignalScore

log = structlog.get_logger(__name__)


class ShadowTextCycleScorer(Protocol):
    def score_cycle(
        self,
        events: list[TextEvent],
        text_contents: dict[uuid.UUID, str],
        strategy_run: StrategyRun,
        as_of: datetime,
        market_prices: dict[uuid.UUID, Decimal] | None = None,
    ) -> Awaitable[list[SignalScore]]: ...


async def run_shadow_text_cycle(
    *,
    scorer: ShadowTextCycleScorer | None,
    session: Session,
    strategy_run: StrategyRun | None,
    as_of: datetime,
    market_prices: Mapping[uuid.UUID, Decimal] | None = None,
) -> None:
    """Run shadow text scoring without affecting the classical engine cycle."""
    if scorer is None or strategy_run is None:
        return

    try:
        if await _should_skip_for_ic_gate(session, strategy_run, as_of):
            return

        window_start = as_of - timedelta(days=1)
        events = await session.text_event_store.get_events(window_start, as_of)
        if not events:
            return

        text_contents: dict[uuid.UUID, str] = {}
        for event in events:
            with contextlib.suppress(OSError):
                text_contents[event.event_id] = Path(event.artifact_uri).read_text(encoding="utf-8")

        market_price_map = dict(market_prices) if market_prices is not None else None
        await scorer.score_cycle(
            events=events,
            text_contents=text_contents,
            strategy_run=strategy_run,
            as_of=as_of,
            market_prices=market_price_map,
        )
    except Exception as exc:
        log.warning(
            "engine_runner.shadow_text_cycle.failed",
            error=str(exc),
            as_of=str(as_of),
        )


async def _should_skip_for_ic_gate(
    session: Session,
    strategy_run: StrategyRun,
    as_of: datetime,
) -> bool:
    perf_repo = getattr(session, "performance_repo", None)
    if perf_repo is None or not hasattr(perf_repo, "status"):
        return False
    try:
        gate = await perf_repo.status(
            strategy_run.strategy_name,
            as_of=as_of,
        )
        warmed_up = gate.observations >= gate.min_observations
        if warmed_up and not gate.passed:
            log.info(
                "engine_runner.text_gate_bypassed",
                strategy=strategy_run.strategy_name,
                rolling_ic=gate.rolling_ic,
                negative_streak=gate.negative_streak,
            )
            return True
    except Exception as gate_exc:
        log.warning(
            "engine_runner.text_gate_check_failed",
            error=str(gate_exc),
        )
    return False
