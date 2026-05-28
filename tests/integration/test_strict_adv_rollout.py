"""Integration tests for strict ADV participation rollout behavior."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.config import (
    LiquiditySettings,
    PlatformSettings,
    RiskSettings,
    StorageSettings,
)
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.events import OrderRejected
from quant_platform.services.data_service.ingest.daily_ingest import refresh_liquidity_from_store
from quant_platform.session import create_paper_session, run_strategy_cycle

_UTC = UTC


def _strategy_run(as_of: datetime) -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="strict_adv_test",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=as_of,
        started_at=as_of,
    )


def _bar(instrument_id: uuid.UUID, ts: datetime, close: Decimal, volume: int) -> MarketBar:
    low = close - Decimal("1")
    high = close + Decimal("1")
    return MarketBar(
        bar_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{instrument_id}:{ts.isoformat()}"),
        instrument_id=instrument_id,
        timestamp=ts,
        bar_seconds=86400,
        open=close,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=close,
        is_complete=True,
    )


@pytest.mark.asyncio
async def test_strict_adv_blocks_without_profile_and_allows_after_refresh(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        liquidity=LiquiditySettings(
            adv_participation_pct=0.05,
            min_adv_usd=1_000_000.0,
            allow_missing_profile=False,
        ),
        storage=StorageSettings(
            object_store_root=str(tmp_path),
        ),
    )
    contracts = {
        instrument_id: {
            "symbol": "AAPL",
            "exchange": "SMART",
            "currency": "USD",
            "sector": "Information Technology",
        }
    }
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("50000"),
        instrument_contracts=contracts,
    )
    await session.broker.connect()
    session.broker.set_market_price(instrument_id, Decimal("100"))  # type: ignore[attr-defined]

    as_of = datetime(2026, 4, 13, 14, 0, tzinfo=_UTC)
    strategy_run = _strategy_run(as_of)
    feature_data = {instrument_id: {"momentum_1m": 1.0}}
    market_prices = {instrument_id: Decimal("100")}

    first = await run_strategy_cycle(
        session=session,
        feature_data=feature_data,
        strategy_run=strategy_run,
        market_prices=market_prices,
        as_of=as_of,
    )
    assert first.submitted_ids == []
    rejects = [event for event in session.event_bus.history if isinstance(event, OrderRejected)]
    assert any("no liquidity profile available" in event.reason.lower() for event in rejects)

    bars = [
        _bar(
            instrument_id,
            as_of - timedelta(days=idx + 1),
            Decimal("100") + Decimal(str(idx)) * Decimal("0.1"),
            1_200_000 + idx * 1000,
        )
        for idx in range(40)
    ]
    await session.bar_store.store_bars(bars)  # type: ignore[attr-defined]
    refreshed = await refresh_liquidity_from_store(
        instruments=session.contract_master.list_active(),
        bar_store=session.bar_store,
        universe_manager=session.universe_manager,
        as_of=as_of,
    )
    assert refreshed >= 1

    second = await run_strategy_cycle(
        session=session,
        feature_data=feature_data,
        strategy_run=strategy_run,
        market_prices=market_prices,
        as_of=as_of + timedelta(minutes=5),
    )
    assert len(second.submitted_ids) > 0
    await session.broker.disconnect()


@pytest.mark.asyncio
async def test_paper_cycle_runs_with_strict_sector_and_seeded_adv(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        liquidity=LiquiditySettings(
            adv_participation_pct=0.05,
            min_adv_usd=1_000_000.0,
            allow_missing_profile=False,
        ),
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.50"),
            max_sector_weight=Decimal("0.80"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.50"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
            require_sector_mapping=True,
        ),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )
    contracts = {
        instrument_id: {
            "symbol": "MSFT",
            "exchange": "SMART",
            "currency": "USD",
            "sector": "Information Technology",
            "adv_shares_20d": 2_000_000,
            "last_close": "100",
        }
    }
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("100000"),
        instrument_contracts=contracts,
    )
    await session.broker.connect()
    session.broker.set_market_price(instrument_id, Decimal("100"))  # type: ignore[attr-defined]

    as_of = datetime(2026, 4, 13, 15, 0, tzinfo=_UTC)
    result = await run_strategy_cycle(
        session=session,
        feature_data={instrument_id: {"momentum_1m": 1.0}},
        strategy_run=_strategy_run(as_of),
        market_prices={instrument_id: Decimal("100")},
        as_of=as_of,
    )

    assert result.rejected == []
    assert len(result.submitted_ids) > 0
    assert len(result.fills) > 0
    await session.broker.disconnect()
