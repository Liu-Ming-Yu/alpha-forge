"""Regression tests for the frequency-bucket split layout on ParquetBarStore.

After the split, the writer routes daily bars to
``bars/<inst>/daily/<year>.parquet`` and non-daily bars to
``bars/<inst>/intraday/<year>.parquet``. The reader checks both that new
file and (for the transition window) the legacy mixed
``bars/<inst>/<year>.parquet``, filtering on ``bar_seconds`` in both cases.
A daily query against a co-mingled file never sees intraday rows, and a
write that lands on an instrument whose ``(inst, year)`` is still in the
legacy layout absorbs the legacy file into the new layout before its own
data is appended.
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
    """Write directly to the pre-split ``bars/<inst>/<year>.parquet`` path.

    Used to simulate fixture data created under the old layout.
    """
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


@pytest.mark.asyncio
async def test_store_bars_routes_to_daily_and_intraday_subdirs(tmp_path: Path) -> None:
    """A mixed write splits across the daily/ and intraday/ subdirectories."""
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()

    await store.store_bars(
        [
            _bar(instrument_id, datetime(2025, 1, 2, tzinfo=_UTC), 86_400, 100.0),
            _bar(instrument_id, datetime(2025, 1, 2, 14, 30, tzinfo=_UTC), 60, 100.05),
            _bar(instrument_id, datetime(2025, 1, 2, 14, 31, tzinfo=_UTC), 60, 100.06),
        ]
    )

    inst_dir = tmp_path / "bars" / str(instrument_id)
    daily_path = inst_dir / "daily" / "2025.parquet"
    intraday_path = inst_dir / "intraday" / "2025.parquet"
    legacy_path = inst_dir / "2025.parquet"

    assert daily_path.exists()
    assert intraday_path.exists()
    assert not legacy_path.exists()

    daily_rows = pq.read_table(daily_path).to_pylist()
    assert {r["bar_seconds"] for r in daily_rows} == {86_400}
    assert len(daily_rows) == 1

    intraday_rows = pq.read_table(intraday_path).to_pylist()
    assert {r["bar_seconds"] for r in intraday_rows} == {60}
    assert len(intraday_rows) == 2


@pytest.mark.asyncio
async def test_get_bars_returns_only_requested_frequency(tmp_path: Path) -> None:
    """A daily ``get_bars`` call must drop intraday rows in the same partition.

    Covers both layouts at once: the write goes through the new split path,
    and the read still has to filter on bar_seconds defensively (the
    intraday subdirectory may itself hold multiple bar_seconds values).
    """
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()

    daily = [
        _bar(instrument_id, datetime(2025, 1, 2, tzinfo=_UTC), 86_400, 100.0),
        _bar(instrument_id, datetime(2025, 1, 3, tzinfo=_UTC), 86_400, 101.0),
    ]
    intraday = [
        _bar(instrument_id, datetime(2025, 1, 2, 14, 30, tzinfo=_UTC), 60, 100.05),
        _bar(instrument_id, datetime(2025, 1, 2, 14, 31, tzinfo=_UTC), 60, 100.06),
        _bar(instrument_id, datetime(2025, 1, 2, 14, 32, tzinfo=_UTC), 60, 100.07),
    ]
    await store.store_bars(daily + intraday)

    got_daily = await store.get_bars(
        instrument_id,
        bar_seconds=86_400,
        start=datetime(2025, 1, 1, tzinfo=_UTC),
        end=datetime(2025, 12, 31, tzinfo=_UTC),
    )
    assert {b.bar_seconds for b in got_daily} == {86_400}
    assert len(got_daily) == len(daily)

    got_intraday = await store.get_bars(
        instrument_id,
        bar_seconds=60,
        start=datetime(2025, 1, 1, tzinfo=_UTC),
        end=datetime(2025, 12, 31, tzinfo=_UTC),
    )
    assert {b.bar_seconds for b in got_intraday} == {60}
    assert len(got_intraday) == len(intraday)


@pytest.mark.asyncio
async def test_dedup_is_per_bar_seconds_not_per_timestamp(tmp_path: Path) -> None:
    """A daily bar and a 00:00-UTC intraday bar may share a timestamp.

    Post-split they live in different files, so dedup is trivially per-file
    -- this also asserts that a second daily write with the same timestamp
    collapses to one row inside the daily file.
    """
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()

    shared_ts = datetime(2025, 1, 2, 0, 0, tzinfo=_UTC)
    await store.store_bars(
        [
            _bar(instrument_id, shared_ts, 86_400, 100.0),
            _bar(instrument_id, shared_ts, 60, 100.05),
        ]
    )
    # Second write of the same daily timestamp with a fresh bar_id must not
    # double the row count (dedup-on-write keeps the latest).
    await store.store_bars([_bar(instrument_id, shared_ts, 86_400, 100.10)])

    daily_path = tmp_path / "bars" / str(instrument_id) / "daily" / "2025.parquet"
    daily_rows = pq.read_table(daily_path).to_pylist()
    daily_keys = [(r["timestamp"], r["bar_seconds"]) for r in daily_rows]
    assert len(daily_keys) == len(set(daily_keys))
    # Most recent daily write wins.
    assert daily_rows[0]["close"] == pytest.approx(100.10)

    intraday_path = tmp_path / "bars" / str(instrument_id) / "intraday" / "2025.parquet"
    intraday_rows = pq.read_table(intraday_path).to_pylist()
    assert len(intraday_rows) == 1
    assert intraday_rows[0]["bar_seconds"] == 60


@pytest.mark.asyncio
async def test_get_bars_reads_legacy_file_when_new_layout_absent(tmp_path: Path) -> None:
    """A legacy ``<year>.parquet`` is still readable through the store.

    Simulates an instrument that has never been written to under the new
    code: only the pre-split mixed file exists on disk. ``get_bars`` must
    pick up its rows and filter on bar_seconds.
    """
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()

    legacy_bars = [
        _bar(instrument_id, datetime(2024, 6, 3, tzinfo=_UTC), 86_400, 100.0),
        _bar(instrument_id, datetime(2024, 6, 4, tzinfo=_UTC), 86_400, 101.0),
        _bar(instrument_id, datetime(2024, 6, 3, 14, 30, tzinfo=_UTC), 60, 100.05),
    ]
    _seed_legacy_partition(tmp_path, instrument_id, 2024, legacy_bars)

    got_daily = await store.get_bars(
        instrument_id,
        bar_seconds=86_400,
        start=datetime(2024, 1, 1, tzinfo=_UTC),
        end=datetime(2024, 12, 31, tzinfo=_UTC),
    )
    assert len(got_daily) == 2
    assert {b.bar_seconds for b in got_daily} == {86_400}

    got_intraday = await store.get_bars(
        instrument_id,
        bar_seconds=60,
        start=datetime(2024, 1, 1, tzinfo=_UTC),
        end=datetime(2024, 12, 31, tzinfo=_UTC),
    )
    assert len(got_intraday) == 1


@pytest.mark.asyncio
async def test_get_bars_merges_legacy_and_new_layouts(tmp_path: Path) -> None:
    """Partial-migration state: legacy file in 2024, new layout in 2025."""
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()

    _seed_legacy_partition(
        tmp_path,
        instrument_id,
        2024,
        [_bar(instrument_id, datetime(2024, 12, 30, tzinfo=_UTC), 86_400, 99.0)],
    )
    await store.store_bars([_bar(instrument_id, datetime(2025, 1, 2, tzinfo=_UTC), 86_400, 100.0)])

    got = await store.get_bars(
        instrument_id,
        bar_seconds=86_400,
        start=datetime(2024, 1, 1, tzinfo=_UTC),
        end=datetime(2025, 12, 31, tzinfo=_UTC),
    )
    assert [b.timestamp.year for b in got] == [2024, 2025]


@pytest.mark.asyncio
async def test_write_auto_migrates_legacy_partition(tmp_path: Path) -> None:
    """A write to ``(inst, year)`` that still has a legacy file absorbs it.

    The next read should see the legacy rows plus the newly-written row,
    the legacy file should be gone, and both new-layout files should exist
    with the right bar_seconds.
    """
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()

    legacy_bars = [
        _bar(instrument_id, datetime(2025, 1, 2, tzinfo=_UTC), 86_400, 100.0),
        _bar(instrument_id, datetime(2025, 1, 2, 14, 30, tzinfo=_UTC), 60, 100.05),
    ]
    _seed_legacy_partition(tmp_path, instrument_id, 2025, legacy_bars)
    legacy_path = tmp_path / "bars" / str(instrument_id) / "2025.parquet"
    assert legacy_path.exists()

    await store.store_bars([_bar(instrument_id, datetime(2025, 1, 3, tzinfo=_UTC), 86_400, 101.0)])

    # Legacy file is gone, new-layout files exist.
    assert not legacy_path.exists()
    daily_path = tmp_path / "bars" / str(instrument_id) / "daily" / "2025.parquet"
    intraday_path = tmp_path / "bars" / str(instrument_id) / "intraday" / "2025.parquet"
    assert daily_path.exists()
    assert intraday_path.exists()

    got_daily = await store.get_bars(
        instrument_id,
        bar_seconds=86_400,
        start=datetime(2025, 1, 1, tzinfo=_UTC),
        end=datetime(2025, 12, 31, tzinfo=_UTC),
    )
    # Two daily rows -- the legacy one and the new write.
    assert [b.timestamp.day for b in got_daily] == [2, 3]

    got_intraday = await store.get_bars(
        instrument_id,
        bar_seconds=60,
        start=datetime(2025, 1, 1, tzinfo=_UTC),
        end=datetime(2025, 12, 31, tzinfo=_UTC),
    )
    assert [b.timestamp.minute for b in got_intraday] == [30]


@pytest.mark.asyncio
async def test_migrate_legacy_partition_is_idempotent(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()

    _seed_legacy_partition(
        tmp_path,
        instrument_id,
        2024,
        [_bar(instrument_id, datetime(2024, 6, 3, tzinfo=_UTC), 86_400, 100.0)],
    )
    store._migrate_legacy_partition_if_present(instrument_id, 2024)  # noqa: SLF001
    # Second call is a no-op (legacy already gone).
    store._migrate_legacy_partition_if_present(instrument_id, 2024)  # noqa: SLF001

    legacy_path = tmp_path / "bars" / str(instrument_id) / "2024.parquet"
    assert not legacy_path.exists()
    daily_path = tmp_path / "bars" / str(instrument_id) / "daily" / "2024.parquet"
    assert daily_path.exists()
    assert len(pq.read_table(daily_path).to_pylist()) == 1
