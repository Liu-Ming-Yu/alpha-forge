"""Vectorized intraday backtest comparator engine."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from quant_platform.config import PlatformSettings
from quant_platform.core.domain.research import (
    IntradayBacktestSpec,
    RunStatus,
    RunType,
    StrategyRun,
)
from quant_platform.services.research_service.backtesting.simple.replay_clock import FakeClock
from quant_platform.services.research_service.backtesting.slippage import IBKRCommissionSchedule
from quant_platform.services.research_service.intraday.backtesting.helpers import (
    assert_feature_availability as _assert_feature_availability,
)
from quant_platform.services.research_service.intraday.backtesting.helpers import (
    max_drawdown as _max_drawdown,
)
from quant_platform.services.research_service.intraday.backtesting.helpers import (
    prices_at as _prices_at,
)
from quant_platform.services.research_service.intraday.backtesting.types import (
    IntradayBacktestResult,
)
from quant_platform.services.research_service.intraday.evidence.evidence import (
    _intraday_input_hash,
    _write_intraday_artifacts,
)
from quant_platform.services.research_service.intraday.vectorized.clone import (
    clone_event_result_as_vectorized,
)
from quant_platform.services.research_service.intraday.vectorized.execution import (
    rebalance_to_target_percent as _rebalance_to_target_percent,
)
from quant_platform.services.research_service.intraday.vectorized.state import (
    VectorizedIntradayReplayState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime
    from pathlib import Path

    from quant_platform.core.contracts import PaperSessionFactory, PortfolioConstructor
    from quant_platform.core.domain.market_data import MarketBar

__all__ = ["VectorizedIntradayBacktestEngine"]


class VectorizedIntradayBacktestEngine:
    """Independent vectorbt-backed approximation used only for fail-closed checks."""

    def __init__(
        self,
        *,
        settings: PlatformSettings | None = None,
        portfolio_constructor: PortfolioConstructor | None = None,
        signal_model: object | None = None,
        commission_schedule: IBKRCommissionSchedule | None = None,
        paper_session_factory: PaperSessionFactory | None = None,
    ) -> None:
        self._settings = settings or PlatformSettings()
        self._portfolio_constructor = portfolio_constructor
        self._signal_model = signal_model
        self._commission = commission_schedule or IBKRCommissionSchedule()
        self._paper_session_factory = paper_session_factory

    async def run(
        self,
        *,
        spec: IntradayBacktestSpec,
        feature_series: Mapping[datetime, Mapping[uuid.UUID, Mapping[str, float]]],
        feature_available_at: Mapping[datetime, datetime],
        minute_bars: Mapping[uuid.UUID, list[MarketBar]],
        instrument_contracts: Mapping[uuid.UUID, dict[str, object]],
        output_root: Path,
        strategy_run: StrategyRun | None = None,
    ) -> IntradayBacktestResult:
        """Run an independent vectorized replay from the same public inputs.

        The event-driven result is intentionally not an input.  This engine
        recomputes signals, targets, and NAV from feature snapshots and minute
        bars so reconciliation can catch event/vectorized drift.
        """
        try:
            import vectorbt as vbt  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "vectorbt is required for VectorizedIntradayBacktestEngine. "
                "Install with: pip install 'quant-platform[backtest]'"
            ) from exc

        _assert_feature_availability(spec, feature_available_at)
        vectorbt_version = str(getattr(vbt, "__version__", "unknown"))
        clock = FakeClock(spec.start)
        strategy_run = strategy_run or StrategyRun(
            run_id=uuid.uuid4(),
            strategy_name=spec.strategy_name,
            strategy_version=spec.strategy_version,
            run_type=RunType.BACKTEST,
            status=RunStatus.RUNNING,
            config_snapshot={"execution_profile": spec.execution_profile, "engine": "vectorbt"},
            created_at=spec.start,
            started_at=spec.start,
        )
        if self._paper_session_factory is None:
            raise RuntimeError(
                "VectorizedIntradayBacktestEngine requires an injected PaperSessionFactory; "
                "construct it through research.backtesting.runtime."
            )
        session = cast(
            "Any",
            self._paper_session_factory(
                settings=self._settings,
                initial_cash=spec.initial_capital,
                strategy_run_id=strategy_run.run_id,
                clock=clock,
                signal_model=self._signal_model,
                portfolio_constructor=self._portfolio_constructor,
                instrument_contracts=dict(instrument_contracts),
            ),
        )

        state = VectorizedIntradayReplayState(settled_cash=spec.initial_capital)

        for decision_time in spec.decision_times:
            clock.set(decision_time)
            state.advance_to(decision_time)
            feature_data = feature_series.get(decision_time, {})
            account = state.account(
                as_of=decision_time,
                minute_bars=minute_bars,
            )
            market_prices = _prices_at(minute_bars, decision_time)
            signals = await session.signal_ctrl.generate(
                feature_data={iid: dict(values) for iid, values in feature_data.items()},
                strategy_run=strategy_run,
                as_of=decision_time,
            )
            regime = await session.regime_detector.detect(decision_time)
            target = await session.portfolio_ctrl.build(
                signals=signals,
                regime=regime,
                account=account,
                limits=session.risk_limits,
            )
            if target is None:
                state.record_nav(decision_time, account.net_asset_value)
                continue
            state.target_weights[decision_time] = dict(target.weights)
            state.eligible[decision_time] = tuple(sorted(feature_data.keys(), key=str))
            rebalance_fills = _rebalance_to_target_percent(
                strategy_run_id=strategy_run.run_id,
                decision_time=decision_time,
                target_weights=target.weights,
                nav=account.net_asset_value,
                market_prices=market_prices,
                minute_bars=minute_bars,
                positions=state.positions,
                settled_cash=state.settled_cash,
                commission_schedule=self._commission,
            )
            for fill in rebalance_fills:
                state.apply_fill(fill)
            end_account = state.account(
                as_of=decision_time,
                minute_bars=minute_bars,
            )
            state.record_nav(decision_time, end_account.net_asset_value)

        if not state.nav_curve:
            state.record_nav(spec.start, spec.initial_capital)
        final_capital = state.nav_curve[-1][1]
        total_return = (final_capital / spec.initial_capital) - Decimal("1")
        max_drawdown = _max_drawdown([nav for _, nav in state.nav_curve])
        artifact_root = output_root / f"{strategy_run.run_id}_vectorized"
        artifact_root.mkdir(parents=True, exist_ok=True)
        paths = _write_intraday_artifacts(
            artifact_root,
            spec=spec,
            strategy_run_id=strategy_run.run_id,
            nav_curve=state.nav_curve,
            fills=state.fills,
            target_weights=state.target_weights,
            eligible_universe=state.eligible,
            final_capital=final_capital,
            total_return=total_return,
            max_drawdown=max_drawdown,
            engine_name="vectorized_intraday_vectorbt",
            engine_version=vectorbt_version,
            input_hash=_intraday_input_hash(spec, feature_series, minute_bars),
            cost_assumptions={
                "commission_schedule": type(self._commission).__name__,
                "fill_style": "target_percent_at_decision_minute",
            },
        )
        return IntradayBacktestResult(
            strategy_run_id=strategy_run.run_id,
            final_capital=final_capital,
            total_return=total_return,
            max_drawdown=max_drawdown,
            nav_curve=tuple(state.nav_curve),
            target_weights=state.target_weights,
            eligible_universe=state.eligible,
            fills=tuple(state.fills),
            residual_order_count=0,
            artifact_root=artifact_root,
            run_summary_uri=paths["run_summary"].resolve().as_uri(),
            execution_quality_uri=paths["execution_quality"].resolve().as_uri(),
            fills_uri=paths["fills"].resolve().as_uri(),
            target_weights_uri=paths["target_weights"].resolve().as_uri(),
        )

    async def run_from_event_result(
        self,
        event_result: IntradayBacktestResult,
        *,
        output_root: Path,
    ) -> IntradayBacktestResult:
        """Produce a deterministic vectorized comparator from canonical targets.

        The comparator intentionally cannot model residual orders.  When the
        canonical replay has residuals, reconciliation marks the run
        non-comparable and fails promotion.
        """
        return clone_event_result_as_vectorized(event_result, output_root=output_root)
