"""Tests for the OHLC-invariant drop logic inside ``run_daily_ingest``."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.ingest.daily_ingest import (
    _filter_invariant_violations,
    run_daily_ingest,
)
from quant_platform.services.data_service.reference.contract_master import ContractMaster
from quant_platform.services.data_service.reference.universe_manager import UniverseManager
from quant_platform.services.data_service.stores.bar_store import InMemoryBarStore


def _inst(symbol: str = "AAPL") -> Instrument:
    return Instrument(
        instrument_id=uuid.uuid4(),
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        exchange="XNAS",
        currency="USD",
    )


def _valid_bar(instrument_id: uuid.UUID, day: date) -> MarketBar:
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
        bar_seconds=86400,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1000,
        is_complete=True,
    )


def _invariant_violating_bar(instrument_id: uuid.UUID, day: date) -> MarketBar:
    """Bypass ``MarketBar.__post_init__`` to forge a bar with high < open.

    Fetcher adapters should never return such bars but defence-in-depth in
    ``_filter_invariant_violations`` means we must still drop them when
    they do slip through.  We use ``object.__setattr__`` because
    ``MarketBar`` is a frozen dataclass.
    """
    bar = _valid_bar(instrument_id, day)
    object.__setattr__(bar, "high", Decimal("50"))  # < open=100
    return bar


def test_filter_drops_invariant_violations() -> None:
    inst = _inst()
    good = _valid_bar(inst.instrument_id, date(2026, 1, 6))
    bad = _invariant_violating_bar(inst.instrument_id, date(2026, 1, 7))

    kept, drops = _filter_invariant_violations(
        [good, bad],
        {inst.instrument_id: inst.symbol},
    )

    assert len(kept) == 1
    assert kept[0].timestamp.date() == date(2026, 1, 6)
    assert drops == {"AAPL": 1}


@pytest.mark.asyncio
async def test_run_daily_ingest_surfaces_drop_count() -> None:
    inst = _inst()
    bars_to_return = [
        _valid_bar(inst.instrument_id, date(2026, 1, 6)),
        _invariant_violating_bar(inst.instrument_id, date(2026, 1, 7)),
    ]

    async def fetcher(instruments: list[Instrument], start: date, end: date) -> list[MarketBar]:
        return bars_to_return

    bar_store = InMemoryBarStore()
    universe = UniverseManager(contract_master=ContractMaster([inst]))

    result = await run_daily_ingest(
        instruments=[inst],
        bar_store=bar_store,
        universe_manager=universe,
        fetcher=fetcher,
        trade_date=date(2026, 1, 10),
        lookback_days=5,
    )

    assert result.bars_fetched == 2
    assert result.bars_stored == 1
    assert result.bars_dropped == 1
    assert result.drops_by_symbol == {"AAPL": 1}
    assert any("ohlc_invariant_drops" in w for w in (result.quality_warnings or []))
