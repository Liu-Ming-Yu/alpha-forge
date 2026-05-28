"""Unit tests for the IB bar fetcher adapter."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.feeds.ib_bar_fetcher import (
    IBBarFetcher,
    _duration_string,
)


def _make_instrument(symbol: str = "AAPL") -> Instrument:
    return Instrument(
        instrument_id=uuid.uuid4(),
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        exchange="XNAS",
        currency="USD",
    )


def _make_bar(
    instrument_id: uuid.UUID,
    day: date,
    close: Decimal = Decimal("100"),
) -> MarketBar:
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
        bar_seconds=86400,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000,
        is_complete=True,
    )


class _FakeBroker:
    def __init__(self, bars_by_instrument: dict[uuid.UUID, list[MarketBar]]) -> None:
        self._bars = bars_by_instrument
        self.calls: list[tuple[uuid.UUID, int, date, str, str]] = []

    async def fetch_historical_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        end_date: date,
        duration: str = "1 D",
        what_to_show: str = "TRADES",
    ) -> list[MarketBar]:
        self.calls.append((instrument_id, bar_seconds, end_date, duration, what_to_show))
        return list(self._bars.get(instrument_id, []))


@pytest.mark.asyncio
async def test_fetcher_filters_bars_outside_window() -> None:
    inst = _make_instrument()
    start = date(2026, 1, 5)
    end = date(2026, 1, 10)
    bars = [
        _make_bar(inst.instrument_id, date(2026, 1, 3)),  # before window
        _make_bar(inst.instrument_id, date(2026, 1, 6)),
        _make_bar(inst.instrument_id, date(2026, 1, 11)),  # after window
    ]
    broker = _FakeBroker({inst.instrument_id: bars})
    fetcher = IBBarFetcher(broker)

    out = await fetcher([inst], start, end)

    assert len(out) == 1
    assert out[0].timestamp.date() == date(2026, 1, 6)
    assert broker.calls[0][2] == end  # end_date passed through
    assert broker.calls[0][3] == "6 D"  # window-derived duration


@pytest.mark.asyncio
async def test_fetcher_survives_instrument_failure() -> None:
    inst_ok = _make_instrument("AAPL")
    inst_bad = _make_instrument("BAD")

    class _FlakyBroker:
        async def fetch_historical_bars(
            self,
            instrument_id: uuid.UUID,
            bar_seconds: int,
            end_date: date,
            duration: str = "1 D",
            what_to_show: str = "TRADES",
        ) -> list[MarketBar]:
            if instrument_id == inst_bad.instrument_id:
                raise RuntimeError("IB timeout")
            return [_make_bar(instrument_id, date(2026, 1, 6))]

    fetcher = IBBarFetcher(_FlakyBroker())
    out = await fetcher([inst_ok, inst_bad], date(2026, 1, 5), date(2026, 1, 10))

    assert len(out) == 1
    assert out[0].instrument_id == inst_ok.instrument_id


def test_duration_string_picks_shortest_token() -> None:
    assert _duration_string(date(2026, 1, 1), date(2026, 1, 1)) == "1 D"
    assert _duration_string(date(2026, 1, 1), date(2026, 1, 30)) == "30 D"
    assert _duration_string(date(2026, 1, 1), date(2026, 3, 1)).endswith(" W")
    assert _duration_string(date(2020, 1, 1), date(2026, 1, 1)).endswith(" Y")
