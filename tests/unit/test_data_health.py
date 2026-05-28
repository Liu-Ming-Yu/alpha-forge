from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.config import LiquiditySettings
from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.reference.contract_master import ContractMaster
from quant_platform.services.data_service.reference.universe_manager import (
    LiquidityProfile,
    UniverseManager,
)
from quant_platform.services.data_service.stores.bar_store import InMemoryBarStore
from quant_platform.services.governance_service.gates.data_health import build_data_health_report

_NOW = datetime(2026, 4, 26, 14, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_data_health_reports_missing_bars_and_liquidity() -> None:
    instrument_id = uuid.uuid4()
    instrument = Instrument(
        instrument_id=instrument_id,
        symbol="AAPL",
        exchange="SMART",
        asset_class=AssetClass.EQUITY,
        currency="USD",
    )
    bar_store = InMemoryBarStore()
    universe = UniverseManager(ContractMaster([instrument]), LiquiditySettings())

    report = await build_data_health_report(
        instruments=[instrument],
        bar_store=bar_store,
        universe_manager=universe,
        start=_NOW,
        end=_NOW,
    )

    assert not report.passed
    assert report.coverage_pct == 0.0
    assert report.statuses[0].issues == (
        "missing_bars",
        "missing_liquidity_profile",
        "stale_bars",
    )


@pytest.mark.asyncio
async def test_data_health_passes_complete_fresh_data() -> None:
    instrument_id = uuid.uuid4()
    instrument = Instrument(
        instrument_id=instrument_id,
        symbol="AAPL",
        exchange="SMART",
        asset_class=AssetClass.EQUITY,
        currency="USD",
    )
    bar_store = InMemoryBarStore()
    await bar_store.store_bars(
        [
            MarketBar(
                bar_id=uuid.uuid4(),
                instrument_id=instrument_id,
                timestamp=_NOW,
                bar_seconds=86400,
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=1000,
            )
        ]
    )
    universe = UniverseManager(ContractMaster([instrument]), LiquiditySettings())
    universe.update_liquidity(
        [
            LiquidityProfile(
                instrument_id=instrument_id,
                adv_shares_20d=1000,
                adv_usd_20d=100_000,
                last_close=Decimal("100"),
                computed_at=_NOW,
            )
        ]
    )

    report = await build_data_health_report(
        instruments=[instrument],
        bar_store=bar_store,
        universe_manager=universe,
        start=_NOW,
        end=_NOW,
    )

    assert report.passed
    assert report.coverage_pct == 1.0
    assert report.liquidity_coverage_pct == 1.0
