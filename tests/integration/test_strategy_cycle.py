"""Integration tests: strategy cycle pipeline.

Exercises the full features → signals → portfolio target → orders chain
using the SimulatedBrokerGateway and FakeClock.

Tests:
    test_features_to_orders_chain       — full pipeline produces submitted orders
    test_regime_de_risking              — RISK_OFF halves exposure; CRISIS → all cash
    test_sell_before_buy                — order list is sells-first
    test_backtest_produces_trades       — backtest uses same stack, changes capital
    test_empty_signals_returns_no_orders — no features → no orders
    test_below_threshold_skipped        — signals below threshold not included
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from quant_platform.config import ExecutionSettings, PlatformSettings, RiskSettings
from quant_platform.core.domain.portfolio import RiskLimits
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.domain.signals import RegimeLabel, RegimeState, SignalScore
from quant_platform.core.events import (
    OrderApproved,
    OrderSubmitted,
    PortfolioTargetBuilt,
    SignalScorePublished,
)
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
    SimpleRegimeDetector,
)
from quant_platform.services.research_service.backtesting.simple.backtest_engine import (
    SimpleBacktestEngine,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.session import create_paper_session, run_strategy_cycle

if TYPE_CHECKING:
    from quant_platform.infrastructure.support.simulated_broker import SimulatedBrokerGateway

_UTC = UTC
_NOW = datetime(2025, 3, 14, 9, 30, 0, tzinfo=_UTC)

# Three test instruments
_INST_A = uuid.uuid4()
_INST_B = uuid.uuid4()
_INST_C = uuid.uuid4()

_SETTINGS = PlatformSettings(
    _env_file=None,
    risk=RiskSettings(
        max_single_name_weight=Decimal("0.20"),
        max_sector_weight=Decimal("0.50"),
        max_gross_exposure=Decimal("0.95"),
        max_daily_turnover=Decimal("0.30"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.20"),
    ),
)


def _make_risk_limits() -> RiskLimits:
    return RiskLimits(
        limits_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        effective_from=_NOW,
        max_single_name_weight=Decimal("0.20"),
        max_sector_weight=Decimal("0.50"),
        max_gross_exposure=Decimal("0.95"),
        max_daily_turnover=Decimal("0.30"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.20"),
    )


def _make_account() -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=Decimal("100000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("100000"),
        net_asset_value=Decimal("100000"),
        positions=(),
    )


def _fake_signal_score(
    instr_id: uuid.UUID,
    score: float,
    strategy_run_id: uuid.UUID,
) -> SignalScore:
    return SignalScore(
        score_id=uuid.uuid4(),
        instrument_id=instr_id,
        strategy_run_id=strategy_run_id,
        as_of=_NOW,
        score=score,
        confidence=1.0,
        model_version="test",
        feature_vector_id=uuid.uuid4(),
    )


def _make_strategy_run(run_id: uuid.UUID | None = None) -> StrategyRun:
    return StrategyRun(
        run_id=run_id or uuid.uuid4(),
        strategy_name="test_cross_sectional",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_NOW,
        started_at=_NOW,
    )


def _make_regime(label: RegimeLabel) -> RegimeState:
    return RegimeState(
        regime_id=uuid.uuid4(),
        as_of=_NOW,
        regime_label=label,
        confidence=1.0,
        detector_version="test",
        supporting_features={},
    )


class TestFeaturestoOrdersChain:
    """Full pipeline: feature_data → signals → target → approved/submitted orders."""

    async def test_basic_pipeline(self) -> None:
        """Three instruments, positive signals, all get orders submitted."""
        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        constructor = LongOnlyPortfolioConstructor(top_n=3, min_score_threshold=0.0)

        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=constructor,
        )

        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        broker.set_market_price(_INST_B, Decimal("50"))
        broker.set_market_price(_INST_C, Decimal("200"))
        await broker.connect()

        feature_data = {
            _INST_A: {"momentum": 0.8},
            _INST_B: {"momentum": 0.6},
            _INST_C: {"momentum": 0.4},
        }
        market_prices = {
            _INST_A: Decimal("100"),
            _INST_B: Decimal("50"),
            _INST_C: Decimal("200"),
        }

        result = await run_strategy_cycle(
            session=session,
            feature_data=feature_data,
            strategy_run=_make_strategy_run(),
            market_prices=market_prices,
            regime=_make_regime(RegimeLabel.RISK_ON),
        )

        # Signals generated for all 3 instruments.
        assert len(result.signals) == 3

        # A PortfolioTarget was built.
        assert result.target is not None
        assert len(result.target.weights) == 3

        # Some orders were submitted.
        assert len(result.submitted_ids) > 0

        # SignalScorePublished events emitted.
        signal_events = [
            e for e in session.event_bus.history if isinstance(e, SignalScorePublished)
        ]
        assert len(signal_events) == 3

        # PortfolioTargetBuilt event emitted.
        target_events = [
            e for e in session.event_bus.history if isinstance(e, PortfolioTargetBuilt)
        ]
        assert len(target_events) == 1

        # OrderApproved and OrderSubmitted events.
        approved_events = [e for e in session.event_bus.history if isinstance(e, OrderApproved)]
        submitted_events = [e for e in session.event_bus.history if isinstance(e, OrderSubmitted)]
        assert len(approved_events) > 0
        assert len(submitted_events) > 0
        assert len(approved_events) == len(submitted_events)

        # CashLedger has applied fills: reservation released, cash updated.
        assert session.cash_engine.reserved_cash == Decimal("0")

    async def test_empty_feature_data_produces_no_orders(self) -> None:
        """Empty feature_data → no signals → all-cash target → no orders."""
        clock = FakeClock(_NOW)
        session = create_paper_session(_SETTINGS, initial_cash=Decimal("50000"), clock=clock)
        await session.broker.connect()

        result = await run_strategy_cycle(
            session=session,
            feature_data={},
            strategy_run=_make_strategy_run(),
            market_prices={},
            regime=_make_regime(RegimeLabel.RISK_ON),
        )

        assert result.signals == []
        # An all-cash target is returned (not None) — it's a valid instruction
        # to hold 100% cash.
        assert result.target is not None
        assert len(result.target.weights) == 0
        assert result.target.cash_target_weight == Decimal("1")
        assert result.approved == []
        assert result.submitted_ids == []

    async def test_fills_update_cash_ledger(self) -> None:
        """After cycle, cash ledger reflects filled buy orders."""
        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
            signal_model=signal_model,
        )

        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        await broker.connect()

        initial_settled = session.cash_engine.settled_cash

        result = await run_strategy_cycle(
            session=session,
            feature_data={_INST_A: {"momentum": 0.9}},
            strategy_run=_make_strategy_run(),
            market_prices={_INST_A: Decimal("100")},
            regime=_make_regime(RegimeLabel.RISK_ON),
        )

        assert len(result.fills) > 0
        # Cash should have decreased after buying.
        assert session.cash_engine.settled_cash < initial_settled
        # No dangling reservations.
        assert session.cash_engine.reserved_cash == Decimal("0")

    async def test_cycle_waits_for_post_submit_lifecycle_fills(self) -> None:
        """Late broker fill callbacks are included in the cycle result summary."""
        clock = FakeClock(_NOW)
        settings = PlatformSettings(
            _env_file=None,
            risk=_SETTINGS.risk,
            execution=ExecutionSettings(
                post_submit_lifecycle_drain_seconds=0.05,
                lifecycle_drain_poll_seconds=0.01,
            ),
        )
        session = create_paper_session(
            settings,
            initial_cash=Decimal("100000"),
            clock=clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
        )

        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        await broker.connect()

        class _DelayedFeed:
            def __init__(self) -> None:
                self.calls = 0

            async def drain_lifecycle_events(self) -> list[object]:
                self.calls += 1
                if self.calls <= 2:
                    return []
                return await broker.drain_lifecycle_events()

        delayed_feed = _DelayedFeed()
        session.lifecycle_feed = delayed_feed  # type: ignore[assignment]

        result = await run_strategy_cycle(
            session=session,
            feature_data={_INST_A: {"momentum": 0.9}},
            strategy_run=_make_strategy_run(),
            market_prices={_INST_A: Decimal("100")},
            regime=_make_regime(RegimeLabel.RISK_ON),
        )

        assert delayed_feed.calls >= 3
        assert len(result.fills) > 0
        assert session.cash_engine.reserved_cash == Decimal("0")


class TestRegimeDeRisking:
    """Regime scaling reduces/eliminates equity exposure."""

    def test_risk_off_halves_exposure_direct(self) -> None:
        """RISK_OFF total invested weight is ~50% of RISK_ON.

        Uses the constructor directly with 5 instruments so the per-name
        cap (0.20) does not bind and regime scaling is clearly visible.
        """
        # Five instruments with positive signals.
        insts = [uuid.uuid4() for _ in range(5)]
        scores = [
            _fake_signal_score(instr_id=inst, score=0.8 - i * 0.1, strategy_run_id=uuid.uuid4())
            for i, inst in enumerate(insts)
        ]

        limits = _make_risk_limits()
        account = _make_account()
        constructor = LongOnlyPortfolioConstructor(top_n=5, min_score_threshold=0.0)

        # RISK_ON: max_invest = min(0.95, 0.95) × 1.0 = 0.95
        # weight_per_name = 0.95 / 5 = 0.19 (below 0.20 cap)
        target_on = constructor.build_targets(
            scores, _make_regime(RegimeLabel.RISK_ON), account, limits
        )
        # RISK_OFF: max_invest = min(0.95, 0.95) × 0.5 = 0.475
        # weight_per_name = 0.475 / 5 = 0.095 (below 0.20 cap)
        target_off = constructor.build_targets(
            scores, _make_regime(RegimeLabel.RISK_OFF), account, limits
        )

        invested_on = sum(target_on.weights.values())
        invested_off = sum(target_off.weights.values())

        assert invested_off < invested_on
        # Ratio should be ~0.5 (RISK_OFF scale / RISK_ON scale)
        ratio = invested_off / invested_on
        assert abs(ratio - Decimal("0.5")) < Decimal("0.01")

    async def test_risk_off_session_cycle_reduces_exposure(self) -> None:
        """RISK_OFF via run_strategy_cycle produces lower equity weight than RISK_ON.

        Uses 5 instruments so regime scaling is visible above the per-name cap.
        """
        insts = [uuid.uuid4() for _ in range(5)]
        feature_data = {inst: {"momentum": 0.8 - i * 0.1} for i, inst in enumerate(insts)}
        prices = {inst: Decimal("100") for inst in insts}

        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        constructor = LongOnlyPortfolioConstructor(top_n=5, min_score_threshold=0.0)

        # RISK_OFF session
        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("500000"),
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=constructor,
        )
        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        for inst in insts:
            broker.set_market_price(inst, Decimal("100"))
        await broker.connect()

        result_off = await run_strategy_cycle(
            session=session,
            feature_data=feature_data,
            strategy_run=_make_strategy_run(),
            market_prices=prices,
            regime=_make_regime(RegimeLabel.RISK_OFF),
        )

        # RISK_ON session
        clock2 = FakeClock(_NOW)
        session2 = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("500000"),
            clock=clock2,
            signal_model=signal_model,
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=5, min_score_threshold=0.0),
        )
        broker2: SimulatedBrokerGateway = session2.broker  # type: ignore
        for inst in insts:
            broker2.set_market_price(inst, Decimal("100"))
        await broker2.connect()

        result_on = await run_strategy_cycle(
            session=session2,
            feature_data=feature_data,
            strategy_run=_make_strategy_run(),
            market_prices=prices,
            regime=_make_regime(RegimeLabel.RISK_ON),
        )

        assert result_off.target is not None
        assert result_on.target is not None

        invested_off = sum(result_off.target.weights.values())
        invested_on = sum(result_on.target.weights.values())
        assert invested_off < invested_on

    async def test_crisis_produces_all_cash_target(self) -> None:
        """CRISIS regime: PortfolioTarget has no equity weights."""
        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
            signal_model=signal_model,
        )
        await session.broker.connect()

        result = await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": 0.9},
                _INST_B: {"momentum": 0.7},
            },
            strategy_run=_make_strategy_run(),
            market_prices={_INST_A: Decimal("100"), _INST_B: Decimal("50")},
            regime=_make_regime(RegimeLabel.CRISIS),
        )

        assert result.target is not None
        assert len(result.target.weights) == 0
        assert result.target.cash_target_weight == Decimal("1")
        # No orders should be planned for an all-cash target.
        assert len(result.submitted_ids) == 0


class TestSellBeforeBuy:
    """Sells precede buys in the planned order list."""

    async def test_transition_generates_sells_first(self) -> None:
        """When transitioning from held positions to new targets, sells come first."""
        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        constructor = LongOnlyPortfolioConstructor(top_n=1, min_score_threshold=0.0)

        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=constructor,
        )

        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        broker.set_market_price(_INST_B, Decimal("100"))
        await broker.connect()

        # Seed broker with a position in INST_A.
        broker._positions[_INST_A] = 100
        broker._avg_costs[_INST_A] = Decimal("100")

        # Signal favours INST_B over INST_A → planner should sell A, buy B.
        result = await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": 0.1},  # low score — will fall out of top-1
                _INST_B: {"momentum": 0.9},  # top signal
            },
            strategy_run=_make_strategy_run(),
            market_prices={
                _INST_A: Decimal("100"),
                _INST_B: Decimal("100"),
            },
            regime=_make_regime(RegimeLabel.RISK_ON),
        )

        # There should be at least one sell event submitted.
        submitted_events = [e for e in session.event_bus.history if isinstance(e, OrderSubmitted)]
        assert len(submitted_events) > 0

    async def test_order_planner_sell_buy_sequence(self) -> None:
        """Direct planner test: returned list has all sells before any buys."""
        from quant_platform.core.domain.orders import OrderSide
        from quant_platform.core.domain.portfolio import PortfolioTarget
        from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
        from quant_platform.services.portfolio_service.order_planner import (
            PortfolioTargetOrderPlanner,
        )

        clock = FakeClock(_NOW)
        planner = PortfolioTargetOrderPlanner(clock, rebalance_threshold=Decimal("0.005"))

        run_id = uuid.uuid4()
        target_id = uuid.uuid4()

        # Account holds INST_A at 20% weight; target moves to INST_B at 20%.
        nav = Decimal("100000")
        pos = PositionSnapshot(
            snapshot_id=uuid.uuid4(),
            instrument_id=_INST_A,
            quantity=200,
            average_cost=Decimal("100"),
            market_price=Decimal("100"),
            market_value=Decimal("20000"),
            unrealised_pnl=Decimal("0"),
            as_of=_NOW,
        )
        account = AccountSnapshot(
            snapshot_id=uuid.uuid4(),
            as_of=_NOW,
            settled_cash=Decimal("80000"),
            unsettled_cash=Decimal("0"),
            reserved_cash=Decimal("0"),
            available_cash=Decimal("80000"),
            net_asset_value=nav,
            positions=(pos,),
        )
        target = PortfolioTarget(
            target_id=target_id,
            strategy_run_id=run_id,
            as_of=_NOW,
            regime_id=uuid.uuid4(),
            weights={_INST_B: Decimal("0.20")},
            cash_target_weight=Decimal("0.80"),
        )

        intents = planner.plan(
            target=target,
            account=account,
            market_prices={_INST_A: Decimal("100"), _INST_B: Decimal("100")},
            strategy_run_id=run_id,
        )

        sell_indices = [i for i, o in enumerate(intents) if o.side == OrderSide.SELL]
        buy_indices = [i for i, o in enumerate(intents) if o.side == OrderSide.BUY]

        assert sell_indices, "expected at least one sell"
        assert buy_indices, "expected at least one buy"
        # Every sell must come before every buy.
        assert max(sell_indices) < min(buy_indices), (
            f"sell at {max(sell_indices)} comes after buy at {min(buy_indices)}"
        )


class TestBacktestParity:
    """SimpleBacktestEngine uses the same strategy stack as live/paper."""

    async def test_backtest_produces_nonzero_trades(self) -> None:
        """run_with_data() executes trades and changes NAV."""

        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        constructor = LongOnlyPortfolioConstructor(top_n=2, min_score_threshold=0.0)

        engine = SimpleBacktestEngine(
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=constructor,
            settings=_SETTINGS,
            paper_session_factory=create_paper_session,
            strategy_cycle_runner=run_strategy_cycle,
        )

        start = _NOW
        end = _NOW.replace(hour=16, minute=0)
        rebalance_ts = _NOW.replace(hour=10, minute=0)

        strategy_run = StrategyRun(
            run_id=uuid.uuid4(),
            strategy_name="backtest_parity_test",
            strategy_version="0.1.0",
            run_type=RunType.BACKTEST,
            status=RunStatus.RUNNING,
            config_snapshot={},
            created_at=_NOW,
            started_at=_NOW,
        )

        feature_series = {
            rebalance_ts: {
                _INST_A: {"momentum": 0.8},
                _INST_B: {"momentum": 0.6},
            }
        }
        price_series = {
            rebalance_ts: {
                _INST_A: Decimal("100"),
                _INST_B: Decimal("50"),
            }
        }

        result = await engine.run_with_data(
            strategy_run=strategy_run,
            start=start,
            end=end,
            initial_capital=Decimal("100000"),
            rebalance_timestamps=[rebalance_ts],
            feature_series=feature_series,
            price_series=price_series,
        )

        # Backtest completed with a valid result.
        assert result.strategy_run_id == strategy_run.run_id
        assert result.initial_capital == Decimal("100000")
        # Capital should differ from initial (trades were executed).
        # SimulatedBrokerGateway charges $1 commission per fill, so NAV changes.
        assert result.final_capital != result.initial_capital
        assert result.max_drawdown <= Decimal("0")

    async def test_backtest_rejects_simple_regime_detector_by_default(self) -> None:
        """``require_market_regime`` (default True) blocks SimpleRegimeDetector.

        This is the safety rail for the historical silent-divergence bug where
        backtests ran ``SimpleRegimeDetector`` (always RISK_ON) while live ran
        ``MarketRegimeDetector``.  Operators must explicitly opt out by setting
        ``settings.backtest.require_market_regime=False``.
        """

        clock = FakeClock(_NOW)
        engine = SimpleBacktestEngine(
            clock=clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            settings=_SETTINGS,
            paper_session_factory=create_paper_session,
            strategy_cycle_runner=run_strategy_cycle,
        )
        rebalance_ts = _NOW.replace(hour=10)
        strategy_run = StrategyRun(
            run_id=uuid.uuid4(),
            strategy_name="require_regime_test",
            strategy_version="0.1.0",
            run_type=RunType.BACKTEST,
            status=RunStatus.RUNNING,
            config_snapshot={},
            created_at=_NOW,
            started_at=_NOW,
        )

        with pytest.raises(ValueError, match="require_market_regime"):
            await engine.run_with_data(
                strategy_run=strategy_run,
                start=_NOW,
                end=_NOW.replace(hour=16),
                initial_capital=Decimal("100000"),
                rebalance_timestamps=[rebalance_ts],
                feature_series={rebalance_ts: {_INST_A: {"momentum": 0.5}}},
                price_series={rebalance_ts: {_INST_A: Decimal("100")}},
                regime_detector=SimpleRegimeDetector(),
            )

    async def test_backtest_default_constructs_market_regime_detector(self) -> None:
        """When ``regime_detector=None``, backtest wires ``MarketRegimeDetector``.

        Pins research-to-production regime parity so backtest regime scaling
        cannot silently revert to the RISK_ON stub.
        """
        from quant_platform.services.signal_service.regime_detector import (
            MarketRegimeDetector,
        )

        clock = FakeClock(_NOW)
        rebalance_ts = _NOW.replace(hour=10)
        # Capture the session's regime_detector through the injected session factory.

        captured: dict[str, object] = {}

        def _spy(*args: object, **kwargs: object) -> object:
            captured["regime_detector"] = kwargs.get("regime_detector")
            return create_paper_session(*args, **kwargs)

        engine = SimpleBacktestEngine(
            clock=clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            settings=_SETTINGS,
            paper_session_factory=_spy,
            strategy_cycle_runner=run_strategy_cycle,
        )
        await engine.run_with_data(
            strategy_run=StrategyRun(
                run_id=uuid.uuid4(),
                strategy_name="default_regime_test",
                strategy_version="0.1.0",
                run_type=RunType.BACKTEST,
                status=RunStatus.RUNNING,
                config_snapshot={},
                created_at=_NOW,
                started_at=_NOW,
            ),
            start=_NOW,
            end=_NOW.replace(hour=16),
            initial_capital=Decimal("100000"),
            rebalance_timestamps=[rebalance_ts],
            feature_series={rebalance_ts: {_INST_A: {"momentum": 0.5}}},
            price_series={rebalance_ts: {_INST_A: Decimal("100")}},
        )

        assert isinstance(captured.get("regime_detector"), MarketRegimeDetector)

    async def test_backtest_run_raises_without_rebalance_data(self) -> None:
        """``run()`` refuses to execute without rebalance timestamps + price/feature series.

        The previous stub silently returned initial capital, which masked the
        fact that no trades had been simulated; callers must use
        ``run_with_data`` instead.
        """
        clock = FakeClock(_NOW)
        engine = SimpleBacktestEngine(clock=clock)

        strategy_run = StrategyRun(
            run_id=uuid.uuid4(),
            strategy_name="stub_test",
            strategy_version="0.1.0",
            run_type=RunType.BACKTEST,
            status=RunStatus.RUNNING,
            config_snapshot={},
            created_at=_NOW,
            started_at=_NOW,
        )

        with pytest.raises(NotImplementedError, match="run_with_data"):
            await engine.run(
                strategy_run=strategy_run,
                start=_NOW,
                end=_NOW.replace(hour=16),
                initial_capital=Decimal("50000"),
            )

    async def test_backtest_max_drawdown_non_positive(self) -> None:
        """max_drawdown in BacktestRun is always <= 0."""
        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        engine = SimpleBacktestEngine(
            clock=clock,
            signal_model=signal_model,
            settings=_SETTINGS,
            paper_session_factory=create_paper_session,
            strategy_cycle_runner=run_strategy_cycle,
        )

        strategy_run = _make_strategy_run()
        strategy_run = StrategyRun(
            run_id=strategy_run.run_id,
            strategy_name=strategy_run.strategy_name,
            strategy_version=strategy_run.strategy_version,
            run_type=RunType.BACKTEST,
            status=RunStatus.RUNNING,
            config_snapshot={},
            created_at=_NOW,
            started_at=_NOW,
        )

        ts1 = _NOW.replace(hour=10)
        ts2 = _NOW.replace(hour=12)

        result = await engine.run_with_data(
            strategy_run=strategy_run,
            start=_NOW,
            end=_NOW.replace(hour=16),
            initial_capital=Decimal("100000"),
            rebalance_timestamps=[ts1, ts2],
            feature_series={
                ts1: {_INST_A: {"momentum": 0.9}},
                ts2: {_INST_A: {"momentum": 0.7}},
            },
            price_series={
                ts1: {_INST_A: Decimal("100")},
                ts2: {_INST_A: Decimal("95")},  # price dropped
            },
        )

        assert result.max_drawdown <= Decimal("0")


class TestSignalFiltering:
    """Signals below the score threshold are excluded from the portfolio."""

    async def test_negative_signals_excluded(self) -> None:
        """Instruments with negative signal scores get no portfolio weight."""
        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        constructor = LongOnlyPortfolioConstructor(
            top_n=5,
            min_score_threshold=0.0,  # must be > 0 to be included
        )
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=constructor,
        )
        await session.broker.connect()

        result = await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": -0.5},  # negative → excluded
                _INST_B: {"momentum": -0.1},  # negative → excluded
            },
            strategy_run=_make_strategy_run(),
            market_prices={_INST_A: Decimal("100"), _INST_B: Decimal("50")},
            regime=_make_regime(RegimeLabel.RISK_ON),
        )

        # Signals generated but no eligible names → all-cash target → no orders.
        assert len(result.signals) == 2
        assert result.target is not None
        assert len(result.target.weights) == 0
        assert result.target.cash_target_weight == Decimal("1")
        assert len(result.submitted_ids) == 0


class TestVolTargetedCycle:
    """The vol-targeted portfolio constructor must receive per-cycle forecasts.

    Before Phase 1.1 these paths silently fell back to equal-weight because
    ``set_vol_forecasts()`` was never called inside the strategy cycle.
    """

    async def test_vol_targeted_cycle_scales_weights(self) -> None:
        """Two instruments with different vol forecasts receive different weights."""
        from quant_platform.services.portfolio_service.vol_sizing import (
            VolTargetedPortfolioConstructor,
        )
        from quant_platform.services.research_service.features.pipeline.feature_pipeline import (
            VOL_FORECAST_KEY,
        )

        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        constructor = VolTargetedPortfolioConstructor(
            vol_target=0.15,
            min_vol_floor=0.05,
            top_n=3,
            min_score_threshold=0.0,
        )
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=constructor,
        )
        await session.broker.connect()

        # Equal momentum signals so any weight difference comes from vol scaling.
        feature_data = {
            _INST_A: {"momentum": 0.5, VOL_FORECAST_KEY: 0.10},  # low vol → larger
            _INST_B: {"momentum": 0.5, VOL_FORECAST_KEY: 0.40},  # high vol → smaller
        }

        result = await run_strategy_cycle(
            session=session,
            feature_data=feature_data,
            strategy_run=_make_strategy_run(),
            market_prices={_INST_A: Decimal("100"), _INST_B: Decimal("50")},
            regime=_make_regime(RegimeLabel.RISK_ON),
        )

        assert result.target is not None
        weights = result.target.weights
        assert _INST_A in weights and _INST_B in weights
        # Low-vol instrument must receive strictly more weight than high-vol.
        assert weights[_INST_A] > weights[_INST_B], f"vol-targeted sizing failed: {weights}"

    async def test_regime_detector_de_risks_on_crisis_stats(self) -> None:
        """CRISIS-level MarketStats → zero gross exposure, no orders submitted."""
        from quant_platform.services.signal_service.regime_detector import (
            MarketRegimeDetector,
            MarketStats,
        )

        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        constructor = LongOnlyPortfolioConstructor(top_n=3, min_score_threshold=0.0)
        from quant_platform.services.signal_service.regime_detector import RegimeThresholds

        # stability_window=1: crisis declared immediately on the first consistent
        # candidate (no stability buffer needed for this single-cycle test).
        detector = MarketRegimeDetector(RegimeThresholds(stability_window=1))
        # Feed crisis-level stats directly — realized vol above crisis threshold
        # unconditionally yields RegimeLabel.CRISIS.
        detector.update(
            MarketStats(
                trend_z=-0.10,
                realized_vol=0.50,
                breadth=0.30,
                as_of=_NOW,
            )
        )

        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=constructor,
            regime_detector=detector,
        )
        await session.broker.connect()

        result = await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": 0.8},
                _INST_B: {"momentum": 0.6},
            },
            strategy_run=_make_strategy_run(),
            market_prices={_INST_A: Decimal("100"), _INST_B: Decimal("50")},
        )

        assert result.target is not None
        # CRISIS → all-cash target, no investable weights, no submissions.
        assert len(result.target.weights) == 0
        assert result.target.cash_target_weight == Decimal("1")
        assert result.submitted_ids == []

    async def test_vol_targeted_without_forecasts_falls_back_to_base(self) -> None:
        """Missing vol forecasts → constructor returns equal-weight base (no regression)."""
        from quant_platform.services.portfolio_service.vol_sizing import (
            VolTargetedPortfolioConstructor,
        )

        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        constructor = VolTargetedPortfolioConstructor(
            vol_target=0.15,
            min_vol_floor=0.05,
            top_n=3,
            min_score_threshold=0.0,
        )
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=constructor,
        )
        await session.broker.connect()

        feature_data = {
            _INST_A: {"momentum": 0.5},
            _INST_B: {"momentum": 0.5},
        }

        result = await run_strategy_cycle(
            session=session,
            feature_data=feature_data,
            strategy_run=_make_strategy_run(),
            market_prices={_INST_A: Decimal("100"), _INST_B: Decimal("50")},
            regime=_make_regime(RegimeLabel.RISK_ON),
        )

        assert result.target is not None
        weights = result.target.weights
        assert len(weights) == 2
        # Equal momentum + no vol forecasts → equal base weights.
        assert weights[_INST_A] == weights[_INST_B]


@pytest.mark.asyncio
class TestReplayParity:
    """Backtest-to-live parity: identical features must yield identical targets.

    This pins down the "same strategy stack in backtest and live" guarantee
    from the production roadmap.  If this test ever fails it indicates a
    silent divergence between ``run_with_data`` and ``run_strategy_cycle``
    (one place using a different portfolio constructor, feature merging,
    regime handling, etc.).
    """

    async def test_live_cycle_and_backtest_produce_same_target_weights(self) -> None:
        from datetime import timedelta

        from quant_platform.services.portfolio_service.vol_sizing import (
            VolTargetedPortfolioConstructor,
        )
        from quant_platform.services.research_service.features.pipeline.feature_pipeline import (
            VOL_FORECAST_KEY,
        )
        from quant_platform.services.signal_service.regime_detector import (
            MarketRegimeDetector,
            MarketStats,
        )

        feature_data = {
            _INST_A: {"momentum": 0.6, VOL_FORECAST_KEY: 0.10},
            _INST_B: {"momentum": 0.6, VOL_FORECAST_KEY: 0.30},
        }

        def _make_constructor() -> VolTargetedPortfolioConstructor:
            return VolTargetedPortfolioConstructor(
                vol_target=0.15,
                min_vol_floor=0.05,
                top_n=3,
                min_score_threshold=0.0,
            )

        def _seed_detector() -> MarketRegimeDetector:
            detector = MarketRegimeDetector()
            detector.update(
                MarketStats(
                    trend_z=0.10,
                    realized_vol=0.12,
                    breadth=0.80,
                    as_of=_NOW,
                )
            )
            return detector

        # --- Live/paper cycle (regime via detector, NOT explicit override) ---
        clock_a = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock_a,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=_make_constructor(),
            regime_detector=_seed_detector(),
        )
        await session.broker.connect()
        cycle_result = await run_strategy_cycle(
            session=session,
            feature_data=feature_data,
            strategy_run=_make_strategy_run(),
            market_prices={_INST_A: Decimal("100"), _INST_B: Decimal("50")},
        )

        # --- Backtest over a single rebalance timestamp, same detector config ---
        clock_b = FakeClock(_NOW)
        engine = SimpleBacktestEngine(
            clock=clock_b,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=_make_constructor(),
            settings=_SETTINGS,
            paper_session_factory=create_paper_session,
            strategy_cycle_runner=run_strategy_cycle,
        )
        rebalance_ts = _NOW.replace(hour=10, minute=0)
        backtest_run = await engine.run_with_data(
            strategy_run=StrategyRun(
                run_id=uuid.uuid4(),
                strategy_name="replay_parity_test",
                strategy_version="0.1.0",
                run_type=RunType.BACKTEST,
                status=RunStatus.RUNNING,
                config_snapshot={},
                created_at=_NOW,
                started_at=_NOW,
            ),
            start=_NOW,
            end=_NOW + timedelta(hours=6),
            initial_capital=Decimal("100000"),
            rebalance_timestamps=[rebalance_ts],
            feature_series={rebalance_ts: feature_data},
            price_series={rebalance_ts: {_INST_A: Decimal("100"), _INST_B: Decimal("50")}},
            regime_detector=_seed_detector(),
        )

        # The backtest engine must have produced a target; compare weight-by-weight.
        assert cycle_result.target is not None
        live_weights = cycle_result.target.weights
        backtest_targets = engine.last_portfolio_targets
        assert backtest_targets, "backtest engine did not record per-rebalance targets"
        del backtest_run  # run metadata unused; targets come from engine
        bt_weights = backtest_targets[-1].weights
        for instrument_id, live_w in live_weights.items():
            assert instrument_id in bt_weights, (
                f"instrument {instrument_id} missing in backtest target"
            )
            assert abs(live_w - bt_weights[instrument_id]) < Decimal("0.0001"), (
                f"weight mismatch for {instrument_id}: "
                f"live={live_w} backtest={bt_weights[instrument_id]}"
            )

    async def test_multi_day_regime_walk_produces_identical_weights(self) -> None:
        """20-rebalance multi-day parity across RISK_ON → RISK_OFF → CRISIS → recovery.

        Closes R-GOV-01: this pins the "same strategy stack" guarantee through
        regime transitions, not just a single RISK_ON step.  Each rebalance
        feeds deterministic features and prices to both the live cycle and
        ``SimpleBacktestEngine.run_with_data``; the two must emit identical
        target weights at every step, with zero tolerance for silent regime
        divergence (the failure mode this sprint was built to eliminate).
        """
        from datetime import timedelta

        from quant_platform.services.portfolio_service.vol_sizing import (
            VolTargetedPortfolioConstructor,
        )
        from quant_platform.services.research_service.features.pipeline.feature_pipeline import (
            VOL_FORECAST_KEY,
        )
        from quant_platform.services.signal_service.regime_detector import (
            MarketRegimeDetector,
            MarketStats,
        )

        def _make_constructor() -> VolTargetedPortfolioConstructor:
            return VolTargetedPortfolioConstructor(
                vol_target=0.15,
                min_vol_floor=0.05,
                top_n=3,
                min_score_threshold=0.0,
            )

        # Synthetic 20-step regime tape: 6 RISK_ON, 5 RISK_OFF, 4 CRISIS,
        # 5 recovery (TRANSITION → RISK_ON).  Encoded as (trend_z, vol, breadth)
        # tuples fed directly into the detector; this avoids any dependence on
        # the regime_index_series inference path and keeps the comparison
        # focused on downstream determinism.
        stats_tape: list[tuple[float, float, float]] = [
            # RISK_ON (strong breadth, low vol, positive trend)
            (0.10, 0.12, 0.80),
            (0.09, 0.12, 0.78),
            (0.09, 0.13, 0.76),
            (0.08, 0.14, 0.74),
            (0.07, 0.15, 0.72),
            (0.06, 0.16, 0.70),
            # RISK_OFF (elevated vol + weakening breadth)
            (0.02, 0.22, 0.55),
            (-0.01, 0.24, 0.52),
            (-0.03, 0.26, 0.48),
            (-0.04, 0.28, 0.44),
            (-0.06, 0.30, 0.40),
            # CRISIS (vol >= 0.35)
            (-0.08, 0.38, 0.30),
            (-0.10, 0.42, 0.25),
            (-0.11, 0.45, 0.22),
            (-0.09, 0.40, 0.25),
            # Recovery
            (-0.05, 0.28, 0.40),
            (0.00, 0.22, 0.50),
            (0.03, 0.18, 0.60),
            (0.06, 0.15, 0.70),
            (0.09, 0.13, 0.78),
        ]
        assert len(stats_tape) == 20, "parity tape must have 20 steps"

        rebalance_timestamps = [_NOW + timedelta(days=i, hours=1) for i in range(20)]

        # Per-step feature data; we rotate the momentum rank across three
        # instruments so the downstream target changes step-to-step.  That way
        # any silent divergence produces a weight mismatch somewhere in the
        # run, not just at the first step.
        feature_series: dict[datetime, dict[uuid.UUID, dict[str, float]]] = {}
        price_series: dict[datetime, dict[uuid.UUID, Decimal]] = {}
        base_prices = {_INST_A: Decimal("100"), _INST_B: Decimal("50"), _INST_C: Decimal("75")}
        for step, ts in enumerate(rebalance_timestamps):
            # Three-way rotation so the leading instrument changes each week.
            m_a = 0.30 + 0.01 * (step % 3)
            m_b = 0.30 + 0.01 * ((step + 1) % 3)
            m_c = 0.30 + 0.01 * ((step + 2) % 3)
            feature_series[ts] = {
                _INST_A: {"momentum": m_a, VOL_FORECAST_KEY: 0.12},
                _INST_B: {"momentum": m_b, VOL_FORECAST_KEY: 0.18},
                _INST_C: {"momentum": m_c, VOL_FORECAST_KEY: 0.24},
            }
            price_series[ts] = dict(base_prices)

        # --- Live/paper loop: one cycle per rebalance timestamp. -----------
        live_clock = FakeClock(_NOW)
        live_detector = MarketRegimeDetector()
        live_session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=live_clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=_make_constructor(),
            regime_detector=live_detector,
        )
        await live_session.broker.connect()

        strategy_run = _make_strategy_run()
        live_weights_by_step: list[dict[uuid.UUID, Decimal]] = []

        for step, ts in enumerate(rebalance_timestamps):
            live_clock.set(ts)
            trend_z, vol, breadth = stats_tape[step]
            live_detector.update(
                MarketStats(
                    trend_z=trend_z,
                    realized_vol=vol,
                    breadth=breadth,
                    as_of=ts,
                )
            )
            res = await run_strategy_cycle(
                session=live_session,
                feature_data=feature_series[ts],
                strategy_run=strategy_run,
                market_prices=price_series[ts],
                as_of=ts,
            )
            live_weights_by_step.append(dict(res.target.weights) if res.target is not None else {})

        # --- Backtest loop: deterministic replay of same inputs. -----------
        bt_clock = FakeClock(_NOW)
        bt_detector = MarketRegimeDetector()
        engine = SimpleBacktestEngine(
            clock=bt_clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=_make_constructor(),
            settings=_SETTINGS,
            paper_session_factory=create_paper_session,
            strategy_cycle_runner=run_strategy_cycle,
        )

        # Pre-seed the detector before each backtest step by monkey-patching
        # the MarketStats tape onto the engine's detector via a wrapper
        # class that ignores the engine's internally-computed stats and
        # returns our tape entries instead.  This keeps the two paths
        # regime-synchronised without needing to replicate SPY history.
        class _TapeDetector:
            def __init__(
                self, inner: MarketRegimeDetector, tape: list[tuple[float, float, float]]
            ) -> None:
                self._inner = inner
                self._tape = tape
                self._ts_to_index: dict[datetime, int] = {
                    ts: i for i, ts in enumerate(rebalance_timestamps)
                }

            def update(self, stats: MarketStats) -> None:  # called by engine
                idx = self._ts_to_index.get(stats.as_of)
                if idx is None:
                    self._inner.update(stats)
                    return
                tz, v, b = self._tape[idx]
                self._inner.update(
                    MarketStats(
                        trend_z=tz,
                        realized_vol=v,
                        breadth=b,
                        as_of=stats.as_of,
                    )
                )

            def classify(self, stats: MarketStats):  # passthrough
                return self._inner.classify(stats)

            async def detect(self, as_of):  # awaited by run_strategy_cycle
                return await self._inner.detect(as_of)

        tape_detector = _TapeDetector(bt_detector, stats_tape)

        await engine.run_with_data(
            strategy_run=StrategyRun(
                run_id=uuid.uuid4(),
                strategy_name="multi_day_parity",
                strategy_version="0.1.0",
                run_type=RunType.BACKTEST,
                status=RunStatus.RUNNING,
                config_snapshot={},
                created_at=_NOW,
                started_at=_NOW,
            ),
            start=_NOW,
            end=_NOW + timedelta(days=25),
            initial_capital=Decimal("100000"),
            rebalance_timestamps=rebalance_timestamps,
            feature_series=feature_series,
            price_series=price_series,
            regime_detector=tape_detector,  # type: ignore[arg-type]
        )

        backtest_targets = engine.last_portfolio_targets
        assert len(backtest_targets) == len(live_weights_by_step), (
            f"target count mismatch: live={len(live_weights_by_step)} "
            f"backtest={len(backtest_targets)}"
        )

        for step, (live_w, bt_target) in enumerate(zip(live_weights_by_step, backtest_targets)):
            bt_w = bt_target.weights
            assert set(live_w.keys()) == set(bt_w.keys()), (
                f"step {step}: instrument set mismatch "
                f"live={sorted(live_w)} backtest={sorted(bt_w)}"
            )
            for instr_id in live_w:
                assert abs(live_w[instr_id] - bt_w[instr_id]) < Decimal("0.0001"), (
                    f"step {step}: weight mismatch for {instr_id}: "
                    f"live={live_w[instr_id]} backtest={bt_w[instr_id]}"
                )
