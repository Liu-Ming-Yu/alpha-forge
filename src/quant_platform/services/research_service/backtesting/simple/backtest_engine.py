"""Concrete BacktestEngine implementation.

SimpleBacktestEngine runs a strategy simulation using the same
signal -> portfolio -> planner -> approve -> submit pipeline as live trading.
The only difference is the BrokerGateway: SimulatedBrokerGateway instead of
IBGatewayBrokerGateway.

Two entry-points:
    run()            - BacktestEngine Protocol compliance; minimal stub that
                       returns initial capital unchanged if no rebalance data
                       is provided.
    run_with_data()  - Full strategy loop; accepts rebalance timestamps,
                       feature series, and price series.  Exercises the
                       complete signal/portfolio/planner stack.

Research-to-production parity:
    run_with_data() calls run_strategy_cycle() from session.py, which is
    identical to the paper/live path.  Only the BrokerGateway is swapped.

Slippage and commission realism:
    When ``slippage_model`` and/or ``commission_schedule`` are provided,
    fills from the SimulatedBrokerGateway are adjusted before position
    accounting.  This prevents over-optimistic backtest results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.config import PlatformSettings
from quant_platform.services.research_service.backtesting.simple.backtest_run_loop import (
    BacktestRunLoopMixin,
)

from ..artifacts.backtest_artifacts import (
    _compute_annualised_sharpe as _annualised_sharpe,
)
from ..simple.backtest_execution_model import (
    CLOSE_AUCTION_SPREAD_MULTIPLIER,
    FALLBACK_ADV_SHARES,
    FALLBACK_SPREAD_BPS,
    STALE_PRICE_BPS,
    BacktestExecutionModel,
    slippage_bps_from_prices,
    spread_bps_for_price,
)
from ..slippage import (
    IBKRCommissionSchedule,
    SlippageModel,
    SquareRootSlippageModel,
)
from .backtest_regime import (
    compute_backtest_market_stats,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.core.contracts import (
        BacktestReplayBroker,
        Clock,
        LiquidityProfileProvider,
        PaperSessionFactory,
        PortfolioConstructor,
        SignalModel,
        StrategyCycleRunner,
    )
    from quant_platform.core.domain.portfolio import PortfolioTarget
    from quant_platform.core.domain.research import (
        BacktestRun,
        StrategyRun,
    )
    from quant_platform.core.regime import MarketStats

log = structlog.get_logger(__name__)

_compute_annualised_sharpe = _annualised_sharpe


class SimpleBacktestEngine(BacktestRunLoopMixin):
    """Backtest engine that runs a strategy over historical rebalance timestamps.

    Uses the same signal/portfolio/planner stack as the live path (research-
    to-production parity).  Only the BrokerGateway is swapped for the
    SimulatedBrokerGateway.

    Args:
        clock: Injectable time source.  A ``FakeClock`` is recommended for
            deterministic tests; ``run_with_data()`` will advance it to each
            rebalance timestamp internally.
        signal_model: SignalModel to use for scoring.  Defaults to
            ``LinearWeightSignalModel({})`` (all-zero scores) if not supplied.
        portfolio_constructor: PortfolioConstructor to use.  Defaults to
            ``LongOnlyPortfolioConstructor()`` with default parameters.
        settings: Platform settings.  Uses defaults when ``None``.
        slippage_model: Model for estimating execution slippage.  Defaults to
            ``SquareRootSlippageModel()`` for realistic market-impact estimation.
        commission_schedule: Commission calculator.  Defaults to
            ``IBKRCommissionSchedule()`` for IBKR tiered pricing.
    """

    # Conservative fallbacks used when a per-instrument LiquidityProfile is
    # not available - deliberately pessimistic so research artifacts never
    # flatter execution versus live.  These constants replace the old
    # hard-coded ``adv_shares=50_000`` / ``spread_bps=5.0`` that applied
    # uniformly to every name regardless of liquidity.
    _FALLBACK_ADV_SHARES = FALLBACK_ADV_SHARES
    _FALLBACK_SPREAD_BPS = FALLBACK_SPREAD_BPS
    _CLOSE_AUCTION_SPREAD_MULTIPLIER = CLOSE_AUCTION_SPREAD_MULTIPLIER
    _STALE_PRICE_BPS = STALE_PRICE_BPS

    def __init__(
        self,
        clock: Clock,
        signal_model: SignalModel | None = None,
        portfolio_constructor: PortfolioConstructor | None = None,
        settings: PlatformSettings | None = None,
        slippage_model: SlippageModel | None = None,
        commission_schedule: IBKRCommissionSchedule | None = None,
        universe_manager: LiquidityProfileProvider | None = None,
        paper_session_factory: PaperSessionFactory | None = None,
        strategy_cycle_runner: StrategyCycleRunner | None = None,
    ) -> None:
        self._clock = clock
        self._signal_model = signal_model
        self._portfolio_constructor = portfolio_constructor
        self._paper_session_factory = paper_session_factory
        self._strategy_cycle_runner = strategy_cycle_runner
        self._settings = settings or PlatformSettings()
        self._slippage_model = slippage_model or SquareRootSlippageModel()
        self._commission_schedule = commission_schedule or IBKRCommissionSchedule()
        self._universe_manager = universe_manager
        self._slippage_fallback_logged: set[uuid.UUID] = set()
        self._execution_model = BacktestExecutionModel(
            settings=self._settings,
            slippage_model=self._slippage_model,
            commission_schedule=self._commission_schedule,
            universe_manager=self._universe_manager,
            fallback_logged=self._slippage_fallback_logged,
        )
        # Per-rebalance PortfolioTargets from the most recent ``run_with_data``
        # call, ordered by rebalance timestamp.  Populated fresh on every run
        # so parity tests (and future analytics) can compare per-step
        # research vs live weights without having to re-thread the targets
        # through the return type.
        self.last_portfolio_targets: list[PortfolioTarget] = []

    def _configure_simulated_execution_model(self, broker: BacktestReplayBroker) -> None:
        """Apply participation, slippage, and commission models to fills.

        Slippage parameters are sourced from ``UniverseManager``'s per-
        instrument ``LiquidityProfile`` when available.  Instruments
        without a profile fall back to the conservative defaults
        (``_FALLBACK_ADV_SHARES``, ``_FALLBACK_SPREAD_BPS``) and emit a
        once-per-instrument log line so operators can triage missing
        liquidity data.
        """
        self._execution_model.configure_simulated_execution_model(broker)

    def _configure_simulated_cost_model(self, broker: BacktestReplayBroker) -> None:
        """Fill cost hook for tests and external callers."""
        self._execution_model.configure_simulated_cost_model(broker)

    def _lookup_liquidity_params(
        self,
        instrument_id: uuid.UUID,
        reference_price: Decimal,
    ) -> tuple[float, float]:
        """Return ``(adv_shares, spread_bps)`` for slippage computation.

        Uses the configured ``UniverseManager`` when available, falling
        back to conservative defaults otherwise.  Logs a one-shot warning
        per missing instrument so repeated rebalances don't spam the
        structured log.
        """
        return self._execution_model.lookup_liquidity_params(instrument_id, reference_price)

    @staticmethod
    def _spread_bps_for_price(reference_price: Decimal) -> float:
        """Heuristic price-bucket spread model (used in the absence of a
        per-sector spread table).

        Wider spreads on sub-$10 tape; tight spreads on mega-caps.  These
        buckets are conservative relative to real IB NBBO tape and can be
        replaced by a sector-table lookup in a follow-up without changing
        the caller interface.
        """
        return spread_bps_for_price(reference_price)

    def _compute_backtest_market_stats(
        self,
        as_of: datetime,
        index_closes: list[float] | None,
        history_closes: dict[uuid.UUID, list[float]],
    ) -> MarketStats | None:
        """Compute MarketStats for the backtest regime detector.

        Preference order:
            1. Caller-provided ``index_closes`` (e.g. real SPY history).
            2. Accumulated price_series snapshots, picking the smallest-UUID
               instrument as a fallback index proxy.

        Returns None when no usable history exists yet; the caller should
        then leave the detector's current state untouched rather than reset it.
        """
        return compute_backtest_market_stats(
            settings=self._settings,
            as_of=as_of,
            index_closes=index_closes,
            history_closes=history_closes,
            fallback_to_price_proxy=True,
            log_event="backtest.regime_stats_failed",
        )

    @staticmethod
    def _slippage_bps_from_prices(
        *,
        side: str,
        model_price: Decimal,
        fill_price: Decimal,
    ) -> float:
        return slippage_bps_from_prices(
            side=side,
            model_price=model_price,
            fill_price=fill_price,
        )

    # ------------------------------------------------------------------
    # BacktestEngine Protocol compliance
    # ------------------------------------------------------------------

    async def run(
        self,
        strategy_run: StrategyRun,
        start: datetime,
        end: datetime,
        initial_capital: Decimal,
    ) -> BacktestRun:
        """Protocol-facing entry-point; intentionally raises to prevent misuse.

        The previous "stub" silently returned ``final_capital == initial_capital``
        which looked like a valid result, masked the fact that no trades had been
        simulated, and encouraged callers to skip ``run_with_data()``.  Raising
        is the only honest behaviour: a backtest without rebalance timestamps
        and price/feature series has not actually been run.

        Callers must use :meth:`run_with_data` and supply rebalance timestamps,
        feature series, and price series to exercise the full strategy loop.
        """
        del strategy_run, start, end, initial_capital  # intentionally unused
        raise NotImplementedError(
            "SimpleBacktestEngine.run() requires rebalance data; "
            "use run_with_data(rebalance_timestamps=..., feature_series=..., "
            "price_series=...) instead."
        )
