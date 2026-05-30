"""Tests for the live MarketBar -> OHLCV-frame adapter (ADR-011 increment 1)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd

from quant_platform.core.domain.market_data.bars import MarketBar
from quant_platform.services.research_service.features.pv_formulaic.bars_frame import (
    OHLCV_FRAME_COLUMNS,
    market_bars_to_ohlcv_frame,
)

_I1 = uuid.UUID("00000000-0000-4000-8000-000000000001")
_I2 = uuid.UUID("00000000-0000-4000-8000-000000000002")


def _bar(instrument_id: uuid.UUID, day: datetime, close: float) -> MarketBar:
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=day,
        bar_seconds=86400,
        open=Decimal(str(close - 0.5)),
        high=Decimal(str(close + 1.0)),
        low=Decimal(str(close - 1.0)),
        close=Decimal(str(close)),
        volume=1000,
    )


def _series(instrument_id: uuid.UUID, start: datetime, closes: list[float]) -> list[MarketBar]:
    return [_bar(instrument_id, start + timedelta(days=i), c) for i, c in enumerate(closes)]


def test_frame_has_contract_columns_and_numeric_types() -> None:
    start = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)
    frame = market_bars_to_ohlcv_frame(
        {_I1: _series(_I1, start, [100.0, 101.0]), _I2: _series(_I2, start, [50.0])}
    )
    assert tuple(frame.columns) == OHLCV_FRAME_COLUMNS
    assert len(frame) == 3
    # OHLCV coerced to float; instrument_id preserved as uuid.UUID.
    for col in ("open", "high", "low", "close", "volume"):
        assert frame[col].dtype == float
    assert all(isinstance(v, uuid.UUID) for v in frame["instrument_id"])


def test_date_is_tz_naive_calendar_day() -> None:
    start = datetime(2026, 3, 2, 21, 30, tzinfo=UTC)
    frame = market_bars_to_ohlcv_frame({_I1: _series(_I1, start, [10.0])})
    only = frame["date"].iloc[0]
    assert isinstance(only, pd.Timestamp)
    assert only.tz is None
    assert (only.hour, only.minute) == (0, 0)
    assert only.date() == datetime(2026, 3, 2).date()


def test_rows_sorted_by_instrument_then_date() -> None:
    start = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)
    # Feed instruments + days out of order; expect sorted output.
    bars = {
        _I2: list(reversed(_series(_I2, start, [50.0, 51.0, 52.0]))),
        _I1: list(reversed(_series(_I1, start, [100.0, 101.0]))),
    }
    frame = market_bars_to_ohlcv_frame(bars)
    inst = list(frame["instrument_id"])
    assert inst == sorted(inst)  # grouped by instrument
    for _, group in frame.groupby("instrument_id", sort=False):
        dates = list(group["date"])
        assert dates == sorted(dates)


def test_as_of_drops_future_bars() -> None:
    start = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)
    frame = market_bars_to_ohlcv_frame(
        {_I1: _series(_I1, start, [100.0, 101.0, 102.0, 103.0])},
        as_of=start + timedelta(days=1, hours=1),  # keep days 0 and 1 only
    )
    assert len(frame) == 2
    assert float(frame["close"].max()) == 101.0


def test_duplicate_instrument_date_keeps_last() -> None:
    day = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)
    frame = market_bars_to_ohlcv_frame(
        {_I1: [_bar(_I1, day, 100.0), _bar(_I1, day, 105.0)]}  # same calendar day
    )
    assert len(frame) == 1
    assert float(frame["close"].iloc[0]) == 105.0


def test_empty_input_returns_empty_frame_with_columns() -> None:
    frame = market_bars_to_ohlcv_frame({})
    assert tuple(frame.columns) == OHLCV_FRAME_COLUMNS
    assert frame.empty
