"""Regression tests for the research-side bar loader.

After the daily/intraday storage split the helper has to read from the
post-split layout AND keep working against legacy mixed files for the
duration of the migration window. Both paths must filter on
``bar_seconds`` so a daily series is never contaminated with intraday
rows regardless of which layout is on disk.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_platform.core.domain.market_data import MarketBar
from quant_platform.research.io.bars import load_bars, load_daily_bars
from quant_platform.services.data_service.stores.parquet_bar_store import ParquetBarStore
from quant_platform.services.data_service.stores.parquet_store_io import BAR_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

_UTC = UTC


def _bar(
    instrument_id: uuid.UUID,
    timestamp: datetime,
    bar_seconds: int,
    close: float,
) -> MarketBar:
    px = Decimal(str(close))
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=timestamp,
        bar_seconds=bar_seconds,
        open=px,
        high=px,
        low=px,
        close=px,
        volume=1_000,
        vwap=None,
        is_complete=True,
    )


def _seed_legacy_partition(
    root: Path,
    instrument_id: uuid.UUID,
    year: int,
    bars: list[MarketBar],
) -> None:
    path = root / "bars" / str(instrument_id) / f"{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "bar_id": [str(b.bar_id) for b in bars],
        "instrument_id": [str(b.instrument_id) for b in bars],
        "timestamp": [b.timestamp for b in bars],
        "bar_seconds": [b.bar_seconds for b in bars],
        "open": [float(b.open) for b in bars],
        "high": [float(b.high) for b in bars],
        "low": [float(b.low) for b in bars],
        "close": [float(b.close) for b in bars],
        "volume": [b.volume for b in bars],
        "vwap": [float(b.vwap) if b.vwap else None for b in bars],
        "is_complete": [b.is_complete for b in bars],
    }
    pq.write_table(pa.table(arrays, schema=BAR_SCHEMA), path)


async def _seed_split_partition(root: Path, instrument_id: uuid.UUID) -> None:
    """Write 2025 daily+intraday bars through the store (post-split layout)."""
    store = ParquetBarStore(root)
    daily = [
        _bar(instrument_id, datetime(2025, 1, 2, tzinfo=_UTC), 86_400, 100.0),
        _bar(instrument_id, datetime(2025, 1, 3, tzinfo=_UTC), 86_400, 101.0),
        _bar(instrument_id, datetime(2025, 1, 6, tzinfo=_UTC), 86_400, 102.0),
    ]
    intraday = [
        _bar(instrument_id, datetime(2025, 1, 2, 14, 30, tzinfo=_UTC), 60, 100.05),
        _bar(instrument_id, datetime(2025, 1, 2, 14, 31, tzinfo=_UTC), 60, 100.06),
        _bar(instrument_id, datetime(2025, 1, 2, 14, 32, tzinfo=_UTC), 60, 100.07),
    ]
    await store.store_bars(daily + intraday)


@pytest.mark.asyncio
async def test_load_daily_bars_from_split_layout(tmp_path: Path) -> None:
    instrument_id = uuid.uuid4()
    await _seed_split_partition(tmp_path, instrument_id)

    df = load_daily_bars(instrument_id, root=tmp_path)

    assert not df.empty
    assert set(df["bar_seconds"].unique().tolist()) == {86_400}
    assert len(df) == 3
    assert df["timestamp"].is_monotonic_increasing


@pytest.mark.asyncio
async def test_load_bars_intraday_from_split_layout(tmp_path: Path) -> None:
    instrument_id = uuid.uuid4()
    await _seed_split_partition(tmp_path, instrument_id)

    df = load_bars(instrument_id, bar_seconds=60, root=tmp_path)

    assert set(df["bar_seconds"].unique().tolist()) == {60}
    assert len(df) == 3


def test_load_daily_bars_from_legacy_mixed_file(tmp_path: Path) -> None:
    """A legacy un-migrated mixed file is still readable with the filter."""
    instrument_id = uuid.uuid4()
    _seed_legacy_partition(
        tmp_path,
        instrument_id,
        2024,
        [
            _bar(instrument_id, datetime(2024, 6, 3, tzinfo=_UTC), 86_400, 100.0),
            _bar(instrument_id, datetime(2024, 6, 4, tzinfo=_UTC), 86_400, 101.0),
            _bar(instrument_id, datetime(2024, 6, 3, 14, 30, tzinfo=_UTC), 60, 100.05),
        ],
    )

    df = load_daily_bars(instrument_id, root=tmp_path)
    assert set(df["bar_seconds"].unique().tolist()) == {86_400}
    assert len(df) == 2


@pytest.mark.asyncio
async def test_load_bars_merges_legacy_and_new_layouts(tmp_path: Path) -> None:
    """Mid-migration: 2024 still in legacy layout, 2025 already split."""
    instrument_id = uuid.uuid4()
    _seed_legacy_partition(
        tmp_path,
        instrument_id,
        2024,
        [_bar(instrument_id, datetime(2024, 12, 30, tzinfo=_UTC), 86_400, 99.0)],
    )
    await _seed_split_partition(tmp_path, instrument_id)

    df = load_daily_bars(instrument_id, root=tmp_path)
    assert len(df) == 4
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].iloc[0] == datetime(2024, 12, 30, tzinfo=_UTC)


@pytest.mark.asyncio
async def test_load_bars_clips_to_window(tmp_path: Path) -> None:
    instrument_id = uuid.uuid4()
    await _seed_split_partition(tmp_path, instrument_id)

    df = load_daily_bars(
        instrument_id,
        root=tmp_path,
        start=datetime(2025, 1, 3, tzinfo=_UTC),
        end=datetime(2025, 1, 5, tzinfo=_UTC),
    )

    assert len(df) == 1
    assert df["timestamp"].iloc[0] == datetime(2025, 1, 3, tzinfo=_UTC)


@pytest.mark.asyncio
async def test_load_bars_with_columns_drops_filter_helpers(tmp_path: Path) -> None:
    """``columns=`` returns only what the caller asked for."""
    instrument_id = uuid.uuid4()
    await _seed_split_partition(tmp_path, instrument_id)

    df = load_daily_bars(
        instrument_id,
        root=tmp_path,
        columns=["timestamp", "close"],
    )

    assert list(df.columns) == ["timestamp", "close"]
    assert len(df) == 3


def test_load_bars_returns_empty_when_instrument_missing(tmp_path: Path) -> None:
    df = load_daily_bars(uuid.uuid4(), root=tmp_path)
    assert df.empty


@pytest.mark.asyncio
async def test_load_bars_returns_empty_when_no_rows_match(tmp_path: Path) -> None:
    instrument_id = uuid.uuid4()
    await _seed_split_partition(tmp_path, instrument_id)

    df = load_daily_bars(
        instrument_id,
        root=tmp_path,
        start=datetime(2030, 1, 1, tzinfo=_UTC),
        end=datetime(2030, 12, 31, tzinfo=_UTC),
    )
    assert df.empty
