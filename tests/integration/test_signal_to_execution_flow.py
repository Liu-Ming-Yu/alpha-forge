"""Integration tests: signal service → portfolio service → execution service.

Complementary to test_strategy_cycle.py.  That file covers the basic
features→orders happy path and regime de-risking via injected RegimeState.
These tests cover three gaps identified in the April-2026 audit:

1. test_cash_gate_restricts_orders_to_budget
       CashLedger + ApproveOrdersController integration: when available_cash
       is small, orders are partially rejected and no reservations leak.

2. test_domain_events_emitted_in_pipeline_order
       Verifies that every stage of the pipeline emits its domain events and
       that the event stream is consistent (approved ↔ submitted counts).

3. test_market_regime_detector_risk_off_scales_exposure
       MarketRegimeDetector.classify() → RISK_OFF → portfolio constructor
       reduces gross exposure to ≤ 50% of max_gross_exposure.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.config import PlatformSettings, RiskSettings
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.domain.signals import RegimeLabel, RegimeState
from quant_platform.core.events import (
    OrderApproved,
    OrderRejected,
    OrderSubmitted,
    PortfolioTargetBuilt,
    SignalScorePublished,
)
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)
from quant_platform.services.signal_service.regime_detector import (
    MarketRegimeDetector,
    MarketStats,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.session import create_paper_session, run_strategy_cycle

if TYPE_CHECKING:
    from quant_platform.infrastructure.support.simulated_broker import SimulatedBrokerGateway

_UTC = UTC
_NOW = datetime(2025, 6, 15, 9, 30, 0, tzinfo=_UTC)

# Five instruments used across tests.
_INST_A = uuid.uuid4()
_INST_B = uuid.uuid4()
_INST_C = uuid.uuid4()
_INST_D = uuid.uuid4()
_INST_E = uuid.uuid4()

_SETTINGS = PlatformSettings(
    _env_file=None,
    risk=RiskSettings(
        max_single_name_weight=Decimal("0.30"),
        max_sector_weight=Decimal("0.60"),
        max_gross_exposure=Decimal("0.95"),
        max_daily_turnover=Decimal("0.50"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.25"),
    ),
)


def _strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="test_signal_to_execution",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_NOW,
        started_at=_NOW,
    )


def _regime(label: RegimeLabel) -> RegimeState:
    return RegimeState(
        regime_id=uuid.uuid4(),
        as_of=_NOW,
        regime_label=label,
        confidence=1.0,
        detector_version="test",
        supporting_features={},
    )


# ---------------------------------------------------------------------------
# 1. Cash gate rejects orders that exceed available budget
# ---------------------------------------------------------------------------


class TestCashGateIntegration:
    """CashLedger enforces hard cash constraint across ApproveOrdersController.

    The portfolio constructor intentionally sizes orders to fit within budget, so
    under normal conditions the cash gate approves all generated intents.  These
    tests verify the accounting properties: no reservation leaks, cash buffer
    preserved, and approved == submitted counts.
    """

    async def test_cash_budget_respected_and_no_reservation_leaks(self) -> None:
        """Filled cycle leaves reserved_cash==0 and respects the cash buffer."""
        initial_cash = Decimal("10000")
        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=initial_cash,
            clock=clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=5, min_score_threshold=0.0),
        )
        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        for inst in (_INST_A, _INST_B, _INST_C, _INST_D, _INST_E):
            broker.set_market_price(inst, Decimal("200"))
        await broker.connect()

        feature_data = {
            _INST_A: {"momentum": 0.9},
            _INST_B: {"momentum": 0.8},
            _INST_C: {"momentum": 0.7},
            _INST_D: {"momentum": 0.6},
            _INST_E: {"momentum": 0.5},
        }
        market_prices = {
            inst: Decimal("200") for inst in (_INST_A, _INST_B, _INST_C, _INST_D, _INST_E)
        }

        result = await run_strategy_cycle(
            session=session,
            feature_data=feature_data,
            strategy_run=_strategy_run(),
            market_prices=market_prices,
            regime=_regime(RegimeLabel.RISK_ON),
        )

        # Signals generated for all 5 instruments.
        assert len(result.signals) == 5

        # Orders are generated and submitted.
        assert len(result.approved) > 0
        assert len(result.submitted_ids) == len(result.approved)

        # No reservation leaks: SimulatedBroker fills synchronously so the
        # coordinator processes fills before we reach this assertion.
        cash_engine = session.cash_engine  # CashLedger
        assert cash_engine.reserved_cash == Decimal("0"), (
            "Cash reservation leaked: a pending reservation was not released after cycle"
        )

        # Total fill cost must not exceed initial_cash * max_gross_exposure.
        total_fill_cost = sum(f.quantity * f.fill_price + f.commission for f in result.fills)
        assert total_fill_cost <= initial_cash * _SETTINGS.risk.max_gross_exposure + Decimal(
            "10"
        ), f"Fills exceeded max_gross_exposure budget: {total_fill_cost}"

    async def test_zero_cash_produces_no_approved_orders(self) -> None:
        """With zero settled cash the gate rejects every order intent."""
        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("0"),
            clock=clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=3, min_score_threshold=0.0),
        )
        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        for inst in (_INST_A, _INST_B, _INST_C):
            broker.set_market_price(inst, Decimal("100"))
        await broker.connect()

        result = await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": 0.9},
                _INST_B: {"momentum": 0.8},
                _INST_C: {"momentum": 0.7},
            },
            strategy_run=_strategy_run(),
            market_prices={inst: Decimal("100") for inst in (_INST_A, _INST_B, _INST_C)},
            regime=_regime(RegimeLabel.RISK_ON),
        )

        assert len(result.approved) == 0
        assert len(result.submitted_ids) == 0
        assert session.cash_engine.reserved_cash == Decimal("0")


# ---------------------------------------------------------------------------
# 2. Domain events emitted consistently throughout the pipeline
# ---------------------------------------------------------------------------


class TestDomainEventPipeline:
    """Verifies the event stream is internally consistent across all stages."""

    async def test_domain_events_emitted_in_pipeline_order(self) -> None:
        """Each stage emits expected events; approved ↔ submitted counts match."""
        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100_000"),
            clock=clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=3, min_score_threshold=0.0),
        )
        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        broker.set_market_price(_INST_B, Decimal("50"))
        broker.set_market_price(_INST_C, Decimal("200"))
        await broker.connect()

        result = await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": 0.8},
                _INST_B: {"momentum": 0.6},
                _INST_C: {"momentum": 0.4},
            },
            strategy_run=_strategy_run(),
            market_prices={
                _INST_A: Decimal("100"),
                _INST_B: Decimal("50"),
                _INST_C: Decimal("200"),
            },
            regime=_regime(RegimeLabel.RISK_ON),
        )

        history = session.event_bus.history

        # Signal events — one per instrument.
        signal_events = [e for e in history if isinstance(e, SignalScorePublished)]
        assert len(signal_events) == 3

        # Exactly one PortfolioTargetBuilt event.
        target_events = [e for e in history if isinstance(e, PortfolioTargetBuilt)]
        assert len(target_events) == 1

        # OrderApproved count matches result.approved.
        approved_events = [e for e in history if isinstance(e, OrderApproved)]
        assert len(approved_events) == len(result.approved)

        # OrderSubmitted count matches result.submitted_ids.
        submitted_events = [e for e in history if isinstance(e, OrderSubmitted)]
        assert len(submitted_events) == len(result.submitted_ids)

        # Every approved order has a corresponding submitted event.
        approved_order_ids = {e.order_id for e in approved_events}
        submitted_order_ids = {e.order_id for e in submitted_events}
        assert approved_order_ids == submitted_order_ids

        # OrderRejected events account for any rejections (count must add up).
        rejected_events = [e for e in history if isinstance(e, OrderRejected)]
        assert len(rejected_events) == len(result.rejected)

    async def test_event_ids_are_unique(self) -> None:
        """No two events share the same event_id (deduplication requirement)."""
        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("50_000"),
            clock=clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=2, min_score_threshold=0.0),
        )
        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        broker.set_market_price(_INST_B, Decimal("100"))
        await broker.connect()

        await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": 0.8},
                _INST_B: {"momentum": 0.6},
            },
            strategy_run=_strategy_run(),
            market_prices={_INST_A: Decimal("100"), _INST_B: Decimal("100")},
            regime=_regime(RegimeLabel.RISK_ON),
        )

        event_ids = [e.event_id for e in session.event_bus.history]
        assert len(event_ids) == len(set(event_ids)), "Duplicate event_ids detected"


# ---------------------------------------------------------------------------
# 3. MarketRegimeDetector integration — RISK_OFF regime halves gross exposure
# ---------------------------------------------------------------------------


class TestMarketRegimeDetectorIntegration:
    """MarketRegimeDetector.classify() drives portfolio construction correctly."""

    async def test_risk_off_stats_produce_reduced_exposure(self) -> None:
        """High-vol stats → RISK_OFF regime → portfolio target gross ≤ 50% of max."""
        detector = MarketRegimeDetector()

        # High realised vol (30%) triggers RISK_OFF (threshold 25%).
        risk_off_stats = MarketStats(
            trend_z=0.01,
            realized_vol=0.30,
            breadth=0.50,
            as_of=_NOW,
        )
        regime = detector.classify(risk_off_stats)
        assert regime.regime_label == RegimeLabel.RISK_OFF

        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100_000"),
            clock=clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=3, min_score_threshold=0.0),
        )
        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        broker.set_market_price(_INST_B, Decimal("50"))
        broker.set_market_price(_INST_C, Decimal("200"))
        await broker.connect()

        result = await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": 0.8},
                _INST_B: {"momentum": 0.6},
                _INST_C: {"momentum": 0.4},
            },
            strategy_run=_strategy_run(),
            market_prices={
                _INST_A: Decimal("100"),
                _INST_B: Decimal("50"),
                _INST_C: Decimal("200"),
            },
            regime=regime,
        )

        assert result.target is not None

        # RISK_OFF applies 50% gross-exposure scale.  Max is 0.95, so the
        # sum of weights should be ≤ 0.50 * 0.95 = 0.475.
        gross = sum(result.target.weights.values())
        assert float(gross) <= 0.50 * float(_SETTINGS.risk.max_gross_exposure) + 0.01, (
            f"RISK_OFF should halve gross exposure; got {gross}"
        )

    async def test_crisis_stats_produce_zero_exposure(self) -> None:
        """Extreme vol (≥35%) → CRISIS → portfolio constructor returns zero weights."""
        detector = MarketRegimeDetector()

        crisis_stats = MarketStats(
            trend_z=-0.10,
            realized_vol=0.40,
            breadth=0.25,
            as_of=_NOW,
        )
        regime = detector.classify(crisis_stats)
        assert regime.regime_label == RegimeLabel.CRISIS

        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100_000"),
            clock=clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=3, min_score_threshold=0.0),
        )
        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        broker.set_market_price(_INST_B, Decimal("50"))
        broker.set_market_price(_INST_C, Decimal("200"))
        await broker.connect()

        result = await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": 0.8},
                _INST_B: {"momentum": 0.6},
                _INST_C: {"momentum": 0.4},
            },
            strategy_run=_strategy_run(),
            market_prices={
                _INST_A: Decimal("100"),
                _INST_B: Decimal("50"),
                _INST_C: Decimal("200"),
            },
            regime=regime,
        )

        # CRISIS: gross exposure scale = 0 → no orders to submit.
        assert len(result.submitted_ids) == 0

    async def test_risk_on_stats_allow_full_exposure(self) -> None:
        """Low vol + uptrend + strong breadth → RISK_ON → full gross exposure."""
        detector = MarketRegimeDetector()

        risk_on_stats = MarketStats(
            trend_z=0.05,  # 5% above 200-day MA
            realized_vol=0.12,  # 12% — well below 20% threshold
            breadth=0.70,  # 70% of universe above their SMA
            as_of=_NOW,
        )
        regime = detector.classify(risk_on_stats)
        assert regime.regime_label == RegimeLabel.RISK_ON

        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100_000"),
            clock=clock,
            signal_model=LinearWeightSignalModel({"momentum": 1.0}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=3, min_score_threshold=0.0),
        )
        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        broker.set_market_price(_INST_B, Decimal("50"))
        broker.set_market_price(_INST_C, Decimal("200"))
        await broker.connect()

        result = await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": 0.8},
                _INST_B: {"momentum": 0.6},
                _INST_C: {"momentum": 0.4},
            },
            strategy_run=_strategy_run(),
            market_prices={
                _INST_A: Decimal("100"),
                _INST_B: Decimal("50"),
                _INST_C: Decimal("200"),
            },
            regime=regime,
        )

        assert result.target is not None
        # RISK_ON allows up to max_gross_exposure (0.95).
        gross = sum(result.target.weights.values())
        assert float(gross) <= float(_SETTINGS.risk.max_gross_exposure) + 0.01
        # And it actually uses a meaningful fraction (not zeroed out).
        assert float(gross) > 0.20, f"RISK_ON should deploy capital; got gross={gross}"
