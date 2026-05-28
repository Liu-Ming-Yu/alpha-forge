"""Walk-forward backtest runner."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.services.research_service.modeling.walk_forward.models import (
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardResult,
)
from quant_platform.services.research_service.modeling.walk_forward.splits import generate_folds

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.research import BacktestRun, StrategyRun
    from quant_platform.services.research_service.backtesting.simple.backtest_engine import (
        SimpleBacktestEngine,
    )

log = structlog.get_logger(__name__)

CalibratorFn = Callable[[WalkForwardFold], Awaitable[dict[str, float] | None]]


class WalkForwardRunner:
    """Drive a rolling-origin evaluation against ``SimpleBacktestEngine``."""

    def __init__(
        self,
        engine: SimpleBacktestEngine,
        calibrator: CalibratorFn | None = None,
    ) -> None:
        self._engine = engine
        self._calibrator = calibrator

    async def run(
        self,
        strategy_run: StrategyRun,
        start: datetime,
        end: datetime,
        initial_capital: Decimal,
        rebalance_timestamps: list[datetime],
        feature_series: dict[datetime, dict[uuid.UUID, dict[str, float]]],
        price_series: dict[datetime, dict[uuid.UUID, Decimal]],
        config: WalkForwardConfig,
    ) -> WalkForwardResult:
        """Run ``engine.run_with_data`` once per evaluated fold."""
        started = strategy_run.created_at
        folds = generate_folds(start, end, config)

        paired: list[tuple[WalkForwardFold, BacktestRun]] = []
        combined = Decimal("1")

        for fold in folds:
            fold_rebalances = [
                ts for ts in rebalance_timestamps if fold.test_start <= ts < fold.test_end
            ]
            if not fold_rebalances:
                log.info(
                    "walk_forward.fold_empty",
                    fold_index=fold.fold_index,
                    test_start=fold.test_start.isoformat(),
                    test_end=fold.test_end.isoformat(),
                )
                continue

            if self._calibrator is not None:
                await self._calibrator(fold)

            fold_features = {
                ts: feats
                for ts, feats in feature_series.items()
                if fold.test_start <= ts < fold.test_end
            }
            fold_prices = {
                ts: prices
                for ts, prices in price_series.items()
                if fold.test_start <= ts < fold.test_end
            }

            run = await self._engine.run_with_data(
                strategy_run=strategy_run,
                start=fold.test_start,
                end=fold.test_end,
                initial_capital=initial_capital,
                rebalance_timestamps=fold_rebalances,
                feature_series=fold_features,
                price_series=fold_prices,
            )
            paired.append((fold, run))
            combined *= Decimal("1") + run.total_return

        finished = strategy_run.finished_at or strategy_run.created_at
        return WalkForwardResult(
            config=config,
            folds=tuple(paired),
            combined_return=combined - Decimal("1"),
            started_at=started,
            finished_at=finished,
            metadata={"requested_folds": len(folds), "evaluated_folds": len(paired)},
        )


__all__ = [
    "CalibratorFn",
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardResult",
    "WalkForwardRunner",
    "generate_folds",
]
