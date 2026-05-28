"""VectorBT-backed backtest engine facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.config import PlatformSettings
from quant_platform.core.algorithms.portfolio_construction import (
    LongOnlyPortfolioConstructor,
    SimpleRegimeDetector,
)

from ..simple.backtest_execution_model import (
    FALLBACK_ADV_SHARES,
    FALLBACK_SPREAD_BPS,
)
from ..simple.backtest_regime import (
    build_backtest_regime_detector,
)
from ..slippage import (
    IBKRCommissionSchedule,
    SlippageModel,
    SquareRootSlippageModel,
)
from .vectorbt_costs import (
    VectorBTCostMixin,
)
from .vectorbt_engine_finalization import (
    finalize_vectorbt_backtest_run,
)
from .vectorbt_engine_runtime import (
    build_vectorbt_runtime,
)
from .vectorbt_regime_series import (
    compute_vectorbt_regime_series,
)
from .vectorbt_signal_frames import (
    build_vectorbt_signal_frames,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    import pandas as pd

    from quant_platform.core.contracts import (
        Clock,
        LiquidityProfileProvider,
        PortfolioConstructor,
        SignalModel,
    )
    from quant_platform.core.domain.research import (
        BacktestRun,
        StrategyRun,
    )
    from quant_platform.core.domain.signals import RegimeState
    from quant_platform.core.regime import (
        MarketRegimeDetector,
    )

    from ..artifacts.backtest_artifacts import (
        BacktestCycleMetrics,
        BacktestFillArtifact,
    )

log = structlog.get_logger(__name__)


class VectorBTBacktestEngine(VectorBTCostMixin):
    """Vectorized backtest engine using VectorBT.

    Implements the ``BacktestEngine`` Protocol.  ``run()`` raises
    ``NotImplementedError``; use ``run_with_data()`` for full batch simulation.

    Preserves identical output schema (``BacktestRun`` with Parquet artifact)
    to ``SimpleBacktestEngine`` so that research and live/paper paths are
    interchangeable at the caller site.
    """

    _FALLBACK_ADV_SHARES = FALLBACK_ADV_SHARES
    _FALLBACK_SPREAD_BPS = FALLBACK_SPREAD_BPS

    def __init__(
        self,
        clock: Clock,
        signal_model: SignalModel | None = None,
        portfolio_constructor: PortfolioConstructor | None = None,
        settings: PlatformSettings | None = None,
        slippage_model: SlippageModel | None = None,
        commission_schedule: IBKRCommissionSchedule | None = None,
        universe_manager: LiquidityProfileProvider | None = None,
        top_n: int = 10,
        parity_mode: bool = False,
    ) -> None:
        self._clock = clock
        self._signal_model = signal_model
        self._portfolio_constructor = portfolio_constructor or LongOnlyPortfolioConstructor()
        self._settings = settings or PlatformSettings()
        self._slippage_model = slippage_model or SquareRootSlippageModel()
        self._commission_schedule = commission_schedule or IBKRCommissionSchedule()
        self._universe_manager = universe_manager
        self._top_n = max(1, top_n)
        # When True, regime scaling is suppressed so results match SimpleBacktestEngine exactly.
        self._parity_mode = parity_mode
        runtime = build_vectorbt_runtime(
            settings=self._settings,
            slippage_model=self._slippage_model,
            commission_schedule=self._commission_schedule,
            universe_manager=self._universe_manager,
            portfolio_constructor=self._portfolio_constructor,
            parity_mode=self._parity_mode,
        )
        self._slippage_fallback_logged = runtime.fallback_logged
        self._execution_model = runtime.execution_model
        self._portfolio_simulator = runtime.portfolio_simulator

    async def run(
        self,
        strategy_run: StrategyRun,
        start: datetime,
        end: datetime,
        initial_capital: Decimal,
    ) -> BacktestRun:
        del strategy_run, start, end, initial_capital
        raise NotImplementedError(
            "VectorBTBacktestEngine.run() requires rebalance data; use run_with_data(...) instead."
        )

    async def run_with_data(
        self,
        strategy_run: StrategyRun,
        start: datetime,
        end: datetime,
        initial_capital: Decimal,
        rebalance_timestamps: list[datetime],
        feature_series: dict[datetime, dict[uuid.UUID, dict[str, float]]],
        price_series: dict[datetime, dict[uuid.UUID, Decimal]],
        regime_detector: MarketRegimeDetector | SimpleRegimeDetector | None = None,
        regime_index_series: dict[datetime, list[float]] | None = None,
    ) -> BacktestRun:
        """Run a vectorized backtest over historical rebalance timestamps.

        Args:
            strategy_run: The StrategyRun record for this backtest.
            start: Simulation start timestamp (timezone-aware).
            end: Simulation end timestamp (timezone-aware).
            initial_capital: Starting cash balance.
            rebalance_timestamps: Ordered list of UTC datetimes at which to
                rebalance.  Each timestamp must be within [start, end].
            feature_series: Mapping of timestamp to feature_data dict.
                feature_data is {instrument_id: {feature_name: float}}.
            price_series: Mapping of timestamp to {instrument_id: price}.
                Instruments absent from a timestamp keep their last known price.
            regime_detector: Optional regime detector.  When ``None`` and
                ``settings.backtest.require_market_regime`` is True (the
                default), a ``MarketRegimeDetector`` is constructed.
            regime_index_series: Optional mapping of timestamp to historical
                close prices for a market proxy (e.g. SPY).

        Returns:
            ``BacktestRun`` with identical schema to ``SimpleBacktestEngine``.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        if end <= start:
            raise ValueError("end must be after start")

        _regime_detector = build_backtest_regime_detector(
            self._settings,
            regime_detector,
            error_suffix=".",
        )

        log.info(
            "vectorbt_backtest.start",
            run_id=str(strategy_run.run_id),
            rebalance_count=len(rebalance_timestamps),
        )

        regime_series = await self._compute_regime_series(
            rebalance_timestamps,
            price_series,
            _regime_detector,
            regime_index_series,
        )

        signal_frames = self.build_signal_frames(
            rebalance_timestamps,
            feature_series,
            price_series,
            strategy_run,
            regime_series,
            top_n=self._top_n,
        )

        nav_curve, fill_artifacts, cycle_metrics = self._simulate_portfolio(
            signal_frames=signal_frames,
            rebalance_timestamps=rebalance_timestamps,
            initial_capital=initial_capital,
            regime_series=regime_series,
        )

        return finalize_vectorbt_backtest_run(
            settings=self._settings,
            clock=self._clock,
            strategy_run=strategy_run,
            start=start,
            end=end,
            initial_capital=initial_capital,
            rebalance_timestamps=rebalance_timestamps,
            nav_curve=nav_curve,
            fill_artifacts=fill_artifacts,
            cycle_metrics=cycle_metrics,
        )

    async def _compute_regime_series(
        self,
        rebalance_timestamps: list[datetime],
        price_series: dict[datetime, dict[uuid.UUID, Decimal]],
        regime_detector: MarketRegimeDetector | SimpleRegimeDetector,
        regime_index_series: dict[datetime, list[float]] | None,
    ) -> dict[datetime, RegimeState]:
        """Pre-compute regime state for each rebalance timestamp.

        Mirrors the logic in ``SimpleBacktestEngine._compute_backtest_market_stats``
        and ``run_strategy_cycle`` update/detect cycle.
        """
        return await compute_vectorbt_regime_series(
            settings=self._settings,
            rebalance_timestamps=rebalance_timestamps,
            price_series=price_series,
            regime_detector=regime_detector,
            regime_index_series=regime_index_series,
        )

    def build_signal_frames(
        self,
        rebalance_timestamps: list[datetime],
        feature_series: dict[datetime, dict[uuid.UUID, dict[str, float]]],
        price_series: dict[datetime, dict[uuid.UUID, Decimal]],
        strategy_run: StrategyRun,
        regime_series: dict[datetime, RegimeState],
        top_n: int = 10,
    ) -> dict[uuid.UUID, pd.DataFrame]:
        """Build per-instrument signal DataFrames.

        Returns dict[instrument_id, DataFrame(index=rebalance_timestamps,
        columns=['close', 'signal', 'regime_scale'])].
        Only the top-N instruments by score (positive scores only) receive
        signal=1.0 at any given timestamp; the rest are 0.0.
        """
        return build_vectorbt_signal_frames(
            rebalance_timestamps=rebalance_timestamps,
            feature_series=feature_series,
            price_series=price_series,
            strategy_run=strategy_run,
            regime_series=regime_series,
            signal_model=self._signal_model,
            portfolio_constructor=self._portfolio_constructor,
            parity_mode=self._parity_mode,
            top_n=top_n,
        )

    def _simulate_portfolio(
        self,
        signal_frames: dict[uuid.UUID, pd.DataFrame],
        rebalance_timestamps: list[datetime],
        initial_capital: Decimal,
        regime_series: dict[datetime, RegimeState],
    ) -> tuple[list[Decimal], list[BacktestFillArtifact], list[BacktestCycleMetrics]]:
        return self._portfolio_simulator.simulate(
            signal_frames=signal_frames,
            rebalance_timestamps=rebalance_timestamps,
            initial_capital=initial_capital,
            regime_series=regime_series,
        )
