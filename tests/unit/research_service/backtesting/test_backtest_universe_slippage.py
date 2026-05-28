"""Tests for universe-aware slippage parameter lookup in
``SimpleBacktestEngine``."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from quant_platform.config import (
    BacktestSettings,
    PlatformSettings,
    RiskSettings,
    StorageSettings,
)
from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.data_service.reference.contract_master import ContractMaster
from quant_platform.services.data_service.reference.universe_manager import (
    LiquidityProfile,
    UniverseManager,
)
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)
from quant_platform.services.research_service.backtesting.simple.backtest_engine import (
    SimpleBacktestEngine,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.session import create_paper_session, run_strategy_cycle

if TYPE_CHECKING:
    from pathlib import Path

_UTC = UTC


def _inst() -> Instrument:
    return Instrument(
        instrument_id=uuid.uuid4(),
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        exchange="XNAS",
        currency="USD",
    )


def _strategy_run() -> StrategyRun:
    now = datetime(2026, 1, 1, tzinfo=_UTC)
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="execution_quality_test",
        strategy_version="0.1.0",
        run_type=RunType.BACKTEST,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=now,
        started_at=now,
    )


def _settings(tmp_path: Path) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        backtest=BacktestSettings(require_market_regime=False),
        risk=RiskSettings(max_single_name_weight=Decimal("0.20")),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )


def _universe(inst: Instrument, *, adv_shares: float) -> UniverseManager:
    universe = UniverseManager(contract_master=ContractMaster([inst]))
    universe.update_liquidity(
        [
            LiquidityProfile(
                instrument_id=inst.instrument_id,
                adv_shares_20d=adv_shares,
                adv_usd_20d=adv_shares * 100.0,
                last_close=Decimal("100"),
                computed_at=datetime(2026, 1, 1, tzinfo=_UTC),
            )
        ]
    )
    return universe


def test_falls_back_to_conservative_params_when_no_profile() -> None:
    engine = SimpleBacktestEngine(clock=FakeClock(datetime(2026, 1, 1, tzinfo=_UTC)))
    adv, spread = engine._lookup_liquidity_params(uuid.uuid4(), Decimal("50"))

    assert adv == SimpleBacktestEngine._FALLBACK_ADV_SHARES
    assert spread == SimpleBacktestEngine._FALLBACK_SPREAD_BPS


def test_uses_profile_adv_when_available() -> None:
    inst = _inst()
    universe = UniverseManager(contract_master=ContractMaster([inst]))
    universe.update_liquidity(
        [
            LiquidityProfile(
                instrument_id=inst.instrument_id,
                adv_shares_20d=1_000_000.0,
                adv_usd_20d=150_000_000.0,
                last_close=Decimal("150"),
                computed_at=datetime(2026, 1, 1, tzinfo=_UTC),
            ),
        ]
    )

    engine = SimpleBacktestEngine(
        clock=FakeClock(datetime(2026, 1, 1, tzinfo=_UTC)),
        universe_manager=universe,
    )
    adv, spread = engine._lookup_liquidity_params(inst.instrument_id, Decimal("150"))

    assert adv == 1_000_000.0
    assert spread == 2.0  # mega-cap bucket


def test_fallback_logged_once_per_instrument() -> None:
    engine = SimpleBacktestEngine(clock=FakeClock(datetime(2026, 1, 1, tzinfo=_UTC)))
    missing_id = uuid.uuid4()

    engine._lookup_liquidity_params(missing_id, Decimal("100"))
    engine._lookup_liquidity_params(missing_id, Decimal("100"))

    assert missing_id in engine._slippage_fallback_logged
    assert len(engine._slippage_fallback_logged) == 1


@pytest.mark.asyncio
async def test_backtest_low_adv_writes_partial_fill_execution_quality(tmp_path: Path) -> None:
    inst = _inst()
    strategy_run = _strategy_run()
    ts = datetime(2026, 1, 2, 15, 0, tzinfo=_UTC)
    engine = SimpleBacktestEngine(
        clock=FakeClock(ts),
        signal_model=LinearWeightSignalModel({"momentum": 1.0}),
        portfolio_constructor=LongOnlyPortfolioConstructor(top_n=1, min_score_threshold=0.0),
        settings=_settings(tmp_path),
        universe_manager=_universe(inst, adv_shares=100.0),
        paper_session_factory=create_paper_session,
        strategy_cycle_runner=run_strategy_cycle,
    )

    await engine.run_with_data(
        strategy_run=strategy_run,
        start=ts,
        end=ts.replace(hour=16),
        initial_capital=Decimal("100000"),
        rebalance_timestamps=[ts],
        feature_series={ts: {inst.instrument_id: {"momentum": 1.0}}},
        price_series={ts: {inst.instrument_id: Decimal("100")}},
    )

    path = tmp_path / "tearsheets" / str(strategy_run.run_id) / "execution_quality.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    order = payload["orders"][0]

    assert aggregate["fill_rate"] < 1.0
    assert order["filled_quantity"] == 5
    assert order["requested_quantity"] > order["filled_quantity"]
    assert order["is_complete"] is False
    assert order["participation_pct"] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_backtest_high_adv_preserves_full_fill_execution_quality(tmp_path: Path) -> None:
    inst = _inst()
    strategy_run = _strategy_run()
    ts = datetime(2026, 1, 2, 15, 0, tzinfo=_UTC)
    engine = SimpleBacktestEngine(
        clock=FakeClock(ts),
        signal_model=LinearWeightSignalModel({"momentum": 1.0}),
        portfolio_constructor=LongOnlyPortfolioConstructor(top_n=1, min_score_threshold=0.0),
        settings=_settings(tmp_path),
        universe_manager=_universe(inst, adv_shares=1_000_000.0),
        paper_session_factory=create_paper_session,
        strategy_cycle_runner=run_strategy_cycle,
    )

    await engine.run_with_data(
        strategy_run=strategy_run,
        start=ts,
        end=ts.replace(hour=16),
        initial_capital=Decimal("100000"),
        rebalance_timestamps=[ts],
        feature_series={ts: {inst.instrument_id: {"momentum": 1.0}}},
        price_series={ts: {inst.instrument_id: Decimal("100")}},
    )

    path = tmp_path / "tearsheets" / str(strategy_run.run_id) / "execution_quality.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    order = payload["orders"][0]

    assert aggregate["fill_rate"] == pytest.approx(1.0)
    assert order["requested_quantity"] == order["filled_quantity"]
    assert order["is_complete"] is True
