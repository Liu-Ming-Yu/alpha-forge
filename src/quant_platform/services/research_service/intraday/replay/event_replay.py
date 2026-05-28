"""Event-driven intraday replay runtime."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from quant_platform.core.domain.research import (
    IntradayBacktestSpec,
    RunStatus,
    RunType,
    StrategyRun,
)
from quant_platform.services.research_service.backtesting.simple.replay_clock import FakeClock
from quant_platform.services.research_service.intraday.backtesting.helpers import (
    account_snapshot,
    advance_settlements,
    apply_fill,
    assert_feature_availability,
    bars_for_order_window,
    prices_at,
    route_execution_tactic,
)
from quant_platform.services.research_service.intraday.backtesting.helpers import (
    max_drawdown as compute_max_drawdown,
)
from quant_platform.services.research_service.intraday.backtesting.types import (
    IntradayBacktestResult,
    IntradayFillArtifact,
)
from quant_platform.services.research_service.intraday.evidence.evidence import (
    _intraday_input_hash,
    _write_intraday_artifacts,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date, datetime
    from pathlib import Path

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import PaperSessionFactory, PortfolioConstructor
    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.core.domain.orders import OrderIntent
    from quant_platform.services.research_service.intraday.replay.replay import (
        IntradayTacticReplayModel,
    )


async def run_event_driven_intraday_backtest(
    *,
    settings: PlatformSettings,
    replay_model: IntradayTacticReplayModel,
    portfolio_constructor: PortfolioConstructor | None,
    signal_model: object | None,
    paper_session_factory: PaperSessionFactory | None = None,
    spec: IntradayBacktestSpec,
    feature_series: Mapping[datetime, Mapping[uuid.UUID, Mapping[str, float]]],
    feature_available_at: Mapping[datetime, datetime],
    minute_bars: Mapping[uuid.UUID, list[MarketBar]],
    instrument_contracts: Mapping[uuid.UUID, dict[str, object]],
    output_root: Path,
    strategy_run: StrategyRun | None = None,
) -> IntradayBacktestResult:
    """Run canonical intraday replay and write evidence artifacts."""
    assert_feature_availability(spec, feature_available_at)
    clock = FakeClock(spec.start)
    strategy_run = strategy_run or StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name=spec.strategy_name,
        strategy_version=spec.strategy_version,
        run_type=RunType.BACKTEST,
        status=RunStatus.RUNNING,
        config_snapshot={"execution_profile": spec.execution_profile},
        created_at=spec.start,
        started_at=spec.start,
    )
    if paper_session_factory is None:
        raise RuntimeError(
            "run_event_driven_intraday_backtest requires an injected PaperSessionFactory; "
            "construct it through research.backtesting.runtime."
        )
    session = cast(
        "Any",
        paper_session_factory(
            settings=settings,
            initial_cash=spec.initial_capital,
            strategy_run_id=strategy_run.run_id,
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=portfolio_constructor,
            instrument_contracts=dict(instrument_contracts),
        ),
    )
    settled_cash = spec.initial_capital
    unsettled_cash = Decimal("0")
    positions: dict[uuid.UUID, int] = {}
    avg_cost: dict[uuid.UUID, Decimal] = {}
    settlements: list[tuple[date, Decimal]] = []
    fills: list[IntradayFillArtifact] = []
    residual_order_count = 0
    target_weights: dict[datetime, Mapping[uuid.UUID, Decimal]] = {}
    eligible: dict[datetime, tuple[uuid.UUID, ...]] = {}
    nav_curve: list[tuple[datetime, Decimal]] = []

    for decision_time in spec.decision_times:
        clock.set(decision_time)
        settled_cash, unsettled_cash, settlements = advance_settlements(
            decision_time.date(), settled_cash, unsettled_cash, settlements
        )
        feature_data = feature_series.get(decision_time, {})
        account = account_snapshot(
            as_of=decision_time,
            settled_cash=settled_cash,
            unsettled_cash=unsettled_cash,
            positions=positions,
            avg_cost=avg_cost,
            minute_bars=minute_bars,
        )
        market_prices = prices_at(minute_bars, decision_time)
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
            nav_curve.append((decision_time, account.net_asset_value))
            continue
        target_weights[decision_time] = dict(target.weights)
        eligible[decision_time] = tuple(sorted(feature_data.keys(), key=str))
        planned = session.order_planner.plan(
            target,
            account,
            market_prices,
            strategy_run.run_id,
        )
        approved: list[OrderIntent] = []
        for intent in planned:
            decision = session.pretrade_gate.evaluate(intent, account, session.risk_limits)
            if decision.passed:
                approved.append(intent)

        for intent in approved:
            route = route_execution_tactic(session.execution_tactic_policy, intent)
            bars = bars_for_order_window(minute_bars.get(intent.instrument_id, []), decision_time)
            order_result = replay_model.replay_order(
                intent,
                bars,
                tactic=route.tactic,
                max_participation_rate=route.max_participation_rate,
                decision_price=market_prices.get(
                    intent.instrument_id, intent.limit_price or Decimal("1")
                ),
            )
            if order_result.residual_quantity > 0:
                residual_order_count += 1
            for fill in order_result.fills:
                settled_cash, unsettled_cash, settlements = apply_fill(
                    fill,
                    settled_cash,
                    unsettled_cash,
                    settlements,
                    positions,
                    avg_cost,
                )
                fills.append(fill)

        end_account = account_snapshot(
            as_of=decision_time,
            settled_cash=settled_cash,
            unsettled_cash=unsettled_cash,
            positions=positions,
            avg_cost=avg_cost,
            minute_bars=minute_bars,
        )
        nav_curve.append((decision_time, end_account.net_asset_value))

    if not nav_curve:
        nav_curve.append((spec.start, spec.initial_capital))
    final_capital = nav_curve[-1][1]
    total_return = (final_capital / spec.initial_capital) - Decimal("1")
    max_drawdown = compute_max_drawdown([nav for _, nav in nav_curve])
    artifact_root = output_root / str(strategy_run.run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    paths = _write_intraday_artifacts(
        artifact_root,
        spec=spec,
        strategy_run_id=strategy_run.run_id,
        nav_curve=nav_curve,
        fills=fills,
        target_weights=target_weights,
        eligible_universe=eligible,
        final_capital=final_capital,
        total_return=total_return,
        max_drawdown=max_drawdown,
        engine_name="event_driven_intraday",
        input_hash=_intraday_input_hash(spec, feature_series, minute_bars),
        cost_assumptions={
            "commission_schedule": "IBKRCommissionSchedule",
            "fill_style": type(replay_model).__name__,
        },
    )
    return IntradayBacktestResult(
        strategy_run_id=strategy_run.run_id,
        final_capital=final_capital,
        total_return=total_return,
        max_drawdown=max_drawdown,
        nav_curve=tuple(nav_curve),
        target_weights=target_weights,
        eligible_universe=eligible,
        fills=tuple(fills),
        residual_order_count=residual_order_count,
        artifact_root=artifact_root,
        run_summary_uri=paths["run_summary"].resolve().as_uri(),
        execution_quality_uri=paths["execution_quality"].resolve().as_uri(),
        fills_uri=paths["fills"].resolve().as_uri(),
        target_weights_uri=paths["target_weights"].resolve().as_uri(),
    )
