"""Mode-specific helper for boosted-tree shadow scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable, Mapping, Sequence
    from datetime import datetime

    from quant_platform.core.domain.research.runs import StrategyRun
    from quant_platform.core.domain.signals import SignalScore


class ShadowBoostingScorer(Protocol):
    def score_cycle(
        self,
        *,
        feature_data: Mapping[uuid.UUID, Mapping[str, float]],
        primary_scores: Sequence[SignalScore],
        strategy_run: StrategyRun,
        as_of: datetime,
    ) -> Awaitable[object | None]: ...


log = structlog.get_logger(__name__)


async def run_shadow_boosting_cycle(
    *,
    scorer: ShadowBoostingScorer | None,
    strategy_run: StrategyRun | None,
    feature_data: dict[uuid.UUID, dict[str, float]],
    primary_scores: Sequence[SignalScore],
    as_of: datetime,
    engine_name: str,
) -> None:
    """Run boosted-tree shadow scoring without changing cycle results."""
    if scorer is None or strategy_run is None:
        return
    try:
        path = await scorer.score_cycle(
            feature_data=feature_data,
            primary_scores=primary_scores,
            strategy_run=strategy_run,
            as_of=as_of,
        )
        if path is not None:
            log.info(
                "engine_runner.shadow_boosting.scored",
                path=str(path),
                instruments=len(feature_data),
            )
    except Exception as exc:
        log.error(
            "engine_runner.shadow_boosting.failed",
            error=str(exc),
            engine=engine_name,
        )
