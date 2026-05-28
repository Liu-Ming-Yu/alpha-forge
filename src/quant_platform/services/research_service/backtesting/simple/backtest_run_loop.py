"""Full replay loop for the simple backtest engine."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from ..artifacts.simple_backtest_cycle_evidence import (
    collect_simple_backtest_cycle_evidence,
)
from .backtest_cycle_guards import (
    assert_features_point_in_time,
    refresh_backtest_regime_detector,
)
from .backtest_regime import (
    BacktestRegimeDetector,
    build_backtest_regime_detector,
)
from .backtest_replay_session import (
    create_backtest_replay_session,
)
from .backtest_run_finalization import (
    finalize_backtest_run,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        BacktestReplayBroker,
        Clock,
        PaperSessionFactory,
        PortfolioConstructor,
        SignalModel,
        StrategyCycleRunner,
    )
    from quant_platform.core.domain.portfolio import PortfolioTarget
    from quant_platform.core.domain.research import BacktestRun, StrategyRun

    from ..artifacts.backtest_artifacts import (
        BacktestCycleMetrics,
        BacktestFillArtifact,
    )


log = structlog.get_logger(__name__)


class BacktestRunLoopMixin:
    """Replay-loop orchestration for ``SimpleBacktestEngine``."""

    _clock: Clock
    _settings: PlatformSettings
    _signal_model: SignalModel | None
    _portfolio_constructor: PortfolioConstructor | None
    _paper_session_factory: PaperSessionFactory | None
    _strategy_cycle_runner: StrategyCycleRunner | None
    last_portfolio_targets: list[PortfolioTarget]

    def _configure_simulated_execution_model(self, broker: BacktestReplayBroker) -> None:
        raise NotImplementedError

    @staticmethod
    def _slippage_bps_from_prices(
        *,
        side: str,
        model_price: Decimal,
        fill_price: Decimal,
    ) -> float:
        raise NotImplementedError

    def _session_factory(self) -> PaperSessionFactory:
        if self._paper_session_factory is not None:
            return self._paper_session_factory
        raise RuntimeError(
            "SimpleBacktestEngine requires an injected PaperSessionFactory; "
            "construct it through research.backtesting.runtime."
        )

    def _cycle_runner(self) -> StrategyCycleRunner:
        if self._strategy_cycle_runner is not None:
            return self._strategy_cycle_runner
        raise RuntimeError(
            "SimpleBacktestEngine requires an injected StrategyCycleRunner; "
            "construct it through research.backtesting.runtime."
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
        regime_detector: BacktestRegimeDetector | None = None,
        regime_index_series: dict[datetime, list[float]] | None = None,
        feature_available_at: dict[datetime, datetime] | None = None,
    ) -> BacktestRun:
        """Full strategy backtest over explicit rebalance timestamps.

        For each timestamp in ``rebalance_timestamps``:
            1. Advance the FakeClock to that timestamp.
            2. Update market prices in the SimulatedBrokerGateway.
            3. Run one strategy cycle (signals -> target -> orders -> fills).

        The same ``run_strategy_cycle()`` function used in paper/live trading
        is called here, ensuring research-to-production parity.

        Args:
            strategy_run: The StrategyRun record for this backtest.
            start: Simulation start timestamp (must be timezone-aware).
            end: Simulation end timestamp (must be timezone-aware).
            initial_capital: Starting cash balance.
            rebalance_timestamps: Ordered list of UTC datetimes at which to
                rebalance.  Each timestamp must be within [start, end].
            feature_series: Mapping of timestamp -> feature_data dict.
                feature_data is {instrument_id: {feature_name: float}}.
                Timestamps not in this dict are skipped (no signals generated).
            price_series: Mapping of timestamp -> {instrument_id: price}.
                Used for share-count calculation and broker price seeding.
                Instruments absent from a timestamp's dict keep their last
                known price.
            regime_detector: Optional RegimeDetector.  When ``None`` and
                ``settings.backtest.require_market_regime`` is True (the
                default), a ``MarketRegimeDetector`` is constructed so backtest
                regime scaling mirrors live.  Passing a ``SimpleRegimeDetector``
                while ``require_market_regime`` is True raises ``ValueError`` -
                this is the safety rail for the historical silent-divergence
                bug where backtests ran RISK_ON while live was regime-aware.
            regime_index_series: Optional mapping of timestamp -> historical
                close prices for a market proxy (e.g. SPY) up to and including
                that rebalance timestamp.  When provided together with a
                ``MarketRegimeDetector``, the engine recomputes ``MarketStats``
                at each rebalance and seeds the detector before the cycle.
                When absent, the engine falls back to accumulating
                ``price_series`` snapshots per instrument and picking the
                smallest-UUID instrument as the index proxy.

        Returns:
            BacktestRun with final capital, total return, and zero-valued
            performance metrics (full risk metrics require a separate
            analytics pass on the trade log, not yet implemented).
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        if end <= start:
            raise ValueError("end must be after start")

        _regime_detector = build_backtest_regime_detector(
            self._settings,
            regime_detector,
        )

        log.info(
            "backtest.run_with_data.start",
            run_id=str(strategy_run.run_id),
            start=str(start),
            end=str(end),
            capital=str(initial_capital),
            rebalance_count=len(rebalance_timestamps),
        )

        # Replay-session construction owns the FakeClock and market-hours override.

        # Backtests replay historical timestamps - forcing market-hours
        # enforcement would reject every order outside 09:30-16:00 NY even
        # for rebalances the live path would have legitimately submitted
        # (which already ran the gate on the original trading day).
        replay = create_backtest_replay_session(
            session_factory=self._session_factory(),
            settings=self._settings,
            start=start,
            initial_capital=initial_capital,
            strategy_run=strategy_run,
            signal_model=self._signal_model,
            portfolio_constructor=self._portfolio_constructor,
            regime_detector=_regime_detector,
        )

        self._configure_simulated_execution_model(replay.broker)
        await replay.broker.connect()

        capital_snapshots: list[Decimal] = [initial_capital]
        cycle_metrics: list[BacktestCycleMetrics] = []
        fill_artifacts: list[BacktestFillArtifact] = []
        self.last_portfolio_targets = []
        cumulative_commission = Decimal("0")
        cumulative_slippage_cost = Decimal("0")
        cumulative_slippage_bps = 0.0

        for ts in rebalance_timestamps:
            if ts < start or ts > end:
                log.warning(
                    "backtest.timestamp_out_of_range",
                    ts=str(ts),
                )
                continue

            replay.fake_clock.set(ts)

            prices_at_ts = price_series.get(ts, {})
            for instr_id, price in prices_at_ts.items():
                replay.broker.set_market_price(instr_id, price)

            # Match the shadow / paper cycle: refresh MarketStats on any
            # detector that exposes ``update`` (MarketRegimeDetector and
            # wrapper classes that compose one - e.g. the parity-test
            # ``_TapeDetector`` that replays a deterministic stats tape).
            # Using a duck-type check instead of ``isinstance`` is
            # consistent with the ``require_market_regime`` guard above,
            # which only rejects ``SimpleRegimeDetector`` explicitly.
            refresh_backtest_regime_detector(
                detector=_regime_detector,
                settings=self._settings,
                as_of=ts,
                price_series_at_ts=prices_at_ts,
                history_closes=replay.history_closes,
                regime_index_series=regime_index_series,
            )

            features_at_ts = feature_series.get(ts, {})
            if not features_at_ts:
                log.debug("backtest.no_features", ts=str(ts))
                continue

            assert_features_point_in_time(
                signal_time=ts,
                feature_available_at=feature_available_at,
            )

            result = await self._cycle_runner()(
                session=replay.session,
                feature_data=features_at_ts,
                strategy_run=strategy_run,
                market_prices=prices_at_ts,
                as_of=ts,
            )

            if result.target is not None:
                self.last_portfolio_targets.append(result.target)

            evidence = await collect_simple_backtest_cycle_evidence(
                ts=ts,
                result=result,
                session=replay.session,
                broker=replay.broker,
                slippage_bps_from_prices=self._slippage_bps_from_prices,
            )
            fill_artifacts.extend(evidence.fill_artifacts)

            cumulative_commission += evidence.commission
            cumulative_slippage_cost += evidence.slippage_cost
            cumulative_slippage_bps += evidence.slippage_bps

            capital_snapshots.append(evidence.nav)
            cycle_metrics.append(evidence.metrics)

            log.debug(
                "backtest.cycle_complete",
                ts=str(ts),
                nav=str(evidence.nav),
                signals=len(result.signals),
                submitted=len(result.submitted_ids),
                fills=len(result.fills),
                commission=str(evidence.commission),
                slippage_cost=str(evidence.slippage_cost),
                slippage_bps=f"{evidence.slippage_bps:.1f}",
            )

        replay.fake_clock.set(end)
        return await finalize_backtest_run(
            settings=self._settings,
            broker=replay.broker,
            strategy_run=strategy_run,
            start=start,
            end=end,
            initial_capital=initial_capital,
            capital_snapshots=capital_snapshots,
            cycle_metrics=cycle_metrics,
            fill_artifacts=fill_artifacts,
            empty_timestamp=self._clock.now(),
            created_at=replay.fake_clock.now(),
            cumulative_commission=cumulative_commission,
            cumulative_slippage_cost=cumulative_slippage_cost,
            cumulative_slippage_bps=cumulative_slippage_bps,
        )
