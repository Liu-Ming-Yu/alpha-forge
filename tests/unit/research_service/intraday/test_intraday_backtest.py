from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from quant_platform.config import BacktestSettings, PlatformSettings, StorageSettings
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.orders import (
    ExecutionTactic,
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.research import IntradayBacktestSpec
from quant_platform.services.research_service.intraday.backtesting.backtest import (
    IntradayBacktestEngine,
    IntradayBacktestResult,
    IntradayTacticReplayModel,
    VectorizedIntradayBacktestEngine,
    reconcile_intraday_backtests,
)
from quant_platform.session import create_paper_session


def _bar(ts: datetime, *, volume: int = 1000) -> MarketBar:
    instrument_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=ts,
        bar_seconds=60,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=volume,
        vwap=Decimal("100"),
    )


def _intent(quantity: int) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        side=OrderSide.BUY,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=datetime(2026, 1, 2, 14, 30, tzinfo=UTC),
        limit_price=Decimal("100"),
    )


def test_intraday_tactic_replay_leaves_residual_when_volume_capacity_is_low() -> None:
    model = IntradayTacticReplayModel()
    bars = [
        _bar(datetime(2026, 1, 2, 14, 30, tzinfo=UTC), volume=1000),
        _bar(datetime(2026, 1, 2, 14, 31, tzinfo=UTC), volume=1000),
    ]

    result = model.replay_order(
        _intent(1000),
        bars,
        tactic=ExecutionTactic.URGENCY_LIMIT,
        max_participation_rate=Decimal("0.05"),
        decision_price=Decimal("100"),
    )

    assert result.residual_quantity > 0
    assert not result.comparable
    assert sum(fill.quantity for fill in result.fills) == 74


def test_intraday_reconciliation_fails_closed_on_residual_orders(tmp_path) -> None:
    run_id = uuid.uuid4()
    now = datetime(2026, 1, 2, 15, 0, tzinfo=UTC)
    common = {
        "strategy_run_id": run_id,
        "final_capital": Decimal("100000"),
        "total_return": Decimal("0"),
        "max_drawdown": Decimal("0"),
        "nav_curve": ((now, Decimal("100000")),),
        "target_weights": {now: {uuid.uuid4(): Decimal("0.10")}},
        "eligible_universe": {now: (uuid.uuid4(),)},
        "fills": (),
        "artifact_root": tmp_path,
        "run_summary_uri": (tmp_path / "run_summary.json").as_uri(),
        "execution_quality_uri": (tmp_path / "execution_quality.json").as_uri(),
        "fills_uri": (tmp_path / "fills.json").as_uri(),
        "target_weights_uri": (tmp_path / "target_weights.json").as_uri(),
    }
    event = IntradayBacktestResult(residual_order_count=1, **common)
    vectorized = IntradayBacktestResult(residual_order_count=0, **common)

    report = reconcile_intraday_backtests(
        event_result=event,
        vectorized_result=vectorized,
        generated_at=now,
    )

    assert not report.passed
    assert not report.comparable
    assert report.status.value == "non_comparable"


@pytest.mark.industrial_backtest
async def test_vectorized_intraday_run_is_independent_from_event_artifacts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vectorbt", SimpleNamespace(__version__="test-vectorbt"))
    instrument_id = uuid.UUID("00000000-0000-0000-0000-000000000011")
    decision_time = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    bars = [
        MarketBar(
            bar_id=uuid.uuid4(),
            instrument_id=instrument_id,
            timestamp=decision_time,
            bar_seconds=60,
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=1_000_000,
            vwap=Decimal("100"),
        )
    ]
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
        backtest=BacktestSettings(require_market_regime=False),
    )
    spec = IntradayBacktestSpec(
        strategy_name="industrial_fixture",
        strategy_version="0.1.0",
        start=decision_time,
        end=decision_time + timedelta(minutes=1),
        initial_capital=Decimal("100000"),
        decision_times=(decision_time,),
        dataset_ids=(uuid.uuid4(),),
        universe_name="fixture",
        feature_set_version="fixture",
        model_version="fixture",
    )
    features = {decision_time: {instrument_id: {"momentum_1m": 1.0}}}
    availability = {decision_time: decision_time}
    contracts = {instrument_id: {"symbol": "AAA", "exchange": "SMART", "currency": "USD"}}

    event_result = await IntradayBacktestEngine(
        settings=settings,
        paper_session_factory=create_paper_session,
    ).run(
        spec=spec,
        feature_series=features,
        feature_available_at=availability,
        minute_bars={instrument_id: bars},
        instrument_contracts=contracts,
        output_root=tmp_path / "event",
    )
    vector_result = await VectorizedIntradayBacktestEngine(
        settings=settings,
        paper_session_factory=create_paper_session,
    ).run(
        spec=spec,
        feature_series=features,
        feature_available_at=availability,
        minute_bars={instrument_id: bars},
        instrument_contracts=contracts,
        output_root=tmp_path / "vector",
    )

    assert vector_result.artifact_root != event_result.artifact_root
    assert vector_result.run_summary_uri != event_result.run_summary_uri
    assert vector_result.target_weights == event_result.target_weights
    summary = (vector_result.artifact_root / "run_summary.json").read_text(encoding="utf-8")
    assert "vectorized_intraday_vectorbt" in summary
