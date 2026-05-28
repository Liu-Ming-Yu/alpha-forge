"""Parquet-backed HistoricalDataStore implementation.

Stores OHLCV bars as Parquet files partitioned by ``(instrument_id, year,
frequency-bucket)``, with corporate-action-adjusted reads.  Uses PyArrow for
efficient columnar I/O.

File layout::

    {root}/
      bars/
        {instrument_id}/
          daily/
            {year}.parquet      # bar_seconds == 86400 rows for one year
          intraday/
            {year}.parquet      # bar_seconds != 86400 rows for one year
          {year}.parquet        # legacy mixed-frequency file (pre-split);
                                # absorbed into the new layout on the next
                                # write to (instrument_id, year), or by the
                                # migrate_bars_split_frequency script.
      corporate_actions/
        {instrument_id}.parquet # append-only corporate action log

The bucket split exists so a research script doing ``pd.read_parquet`` on a
single file cannot accidentally combine daily and intraday rows (which used
to happen in 2025/2026 yearly files for ~15 instruments and silently
contaminate forward-return series). Reads always check both the new-layout
file for the requested bucket and the legacy mixed file (if it still
exists), so the migration can run lazily without breaking historical reads.

Bar deduplication is enforced via bar_id within a partition: writing a bar
whose bar_id already exists in the partition is a no-op. A secondary dedup
on ``(timestamp, bar_seconds)`` runs after merge so two ingest sources
writing the same logical bar with different bar_ids do not both survive.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import structlog

from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.reference.corporate_actions import apply_adjustments
from quant_platform.services.data_service.stores.parquet_corporate_actions import (
    ParquetCorporateActionsMixin,
)
from quant_platform.services.data_service.stores.parquet_store_io import (
    BAR_SCHEMA as _BAR_SCHEMA,
)
from quant_platform.services.data_service.stores.parquet_store_io import (
    atomic_write_table as _atomic_write_table,
)
from quant_platform.services.data_service.stores.parquet_store_io import (
    exclusive_file_lock as _exclusive_file_lock,
)

if TYPE_CHECKING:
    from datetime import datetime

log = structlog.get_logger(__name__)

DAILY_BAR_SECONDS = 86_400
DAILY_BUCKET = "daily"
INTRADAY_BUCKET = "intraday"


def bucket_for(bar_seconds: int) -> str:
    """Frequency-bucket subdirectory for a given ``bar_seconds`` value.

    Daily bars (``86400``) live under ``daily/``; everything else lives under
    ``intraday/``. The split exists to keep raw ``pd.read_parquet`` calls
    from silently mixing frequencies.
    """
    return DAILY_BUCKET if bar_seconds == DAILY_BAR_SECONDS else INTRADAY_BUCKET


class ParquetBarStore(ParquetCorporateActionsMixin):
    """Durable, Parquet-backed implementation of the HistoricalDataStore contract.

    Args:
        root: Base directory for all Parquet files.  Created on first write.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _bar_path(self, instrument_id: uuid.UUID, year: int, bar_seconds: int) -> Path:
        return (
            self._root / "bars" / str(instrument_id) / bucket_for(bar_seconds) / f"{year}.parquet"
        )

    def _legacy_bar_path(self, instrument_id: uuid.UUID, year: int) -> Path:
        """Path to the pre-split mixed-frequency file for one (inst, year)."""
        return self._root / "bars" / str(instrument_id) / f"{year}.parquet"

    def _ca_path(self, instrument_id: uuid.UUID) -> Path:
        return self._root / "corporate_actions" / f"{instrument_id}.parquet"

    def _adjusted_bar_path(self, instrument_id: uuid.UUID, year: int) -> Path:
        return self._root / "bars_adjusted" / str(instrument_id) / f"{year}.parquet"

    def _migrate_legacy_partition_if_present(
        self,
        instrument_id: uuid.UUID,
        year: int,
    ) -> None:
        """Absorb a legacy ``bars/<inst>/<year>.parquet`` into the new layout.

        Idempotent and concurrency-safe via the legacy file's exclusive
        lock: a second caller that runs after the first will see no legacy
        file and return immediately. The legacy file is unlinked only after
        every row has been merged into the appropriate ``daily/`` or
        ``intraday/`` partition.
        """
        legacy_path = self._legacy_bar_path(instrument_id, year)
        if not legacy_path.exists():
            return

        with _exclusive_file_lock(legacy_path):
            # Re-check under the lock: another writer may have completed
            # the migration between the existence check and the lock.
            if not legacy_path.exists():
                return

            legacy_table = pq.read_table(legacy_path, schema=_BAR_SCHEMA)
            if legacy_table.num_rows == 0:
                legacy_path.unlink(missing_ok=True)
                return

            # Split by bucket. ``bar_seconds`` is int32; ``pc.equal`` works
            # against the python int directly.
            bs_col = legacy_table.column("bar_seconds")
            daily_mask = pc.equal(bs_col, DAILY_BAR_SECONDS)
            daily_rows = legacy_table.filter(daily_mask)
            intraday_rows = legacy_table.filter(pc.invert(daily_mask))

            for bucket_table, sample_bs in (
                (daily_rows, DAILY_BAR_SECONDS),
                (intraday_rows, 60),  # any non-daily value picks the intraday bucket
            ):
                if bucket_table.num_rows == 0:
                    continue
                self._merge_into_bucket_file(
                    self._bar_path(instrument_id, year, sample_bs),
                    bucket_table,
                )

            legacy_path.unlink(missing_ok=True)
            log.info(
                "parquet_bar_store.migrated_legacy_partition",
                instrument_id=str(instrument_id),
                year=year,
                daily_rows=daily_rows.num_rows,
                intraday_rows=intraday_rows.num_rows,
            )

    def _merge_into_bucket_file(self, path: Path, new_table: pa.Table) -> None:
        """Read-merge-write ``new_table`` into the parquet at ``path``.

        Holds an exclusive lock on ``path`` for the duration. Applies the
        ``bar_id`` and ``(timestamp, bar_seconds)`` dedups that
        :meth:`_sync_write_partition` also applies. Used both by the
        ordinary write path and by :meth:`_migrate_legacy_partition_if_present`.
        """
        if new_table.num_rows == 0:
            return
        with _exclusive_file_lock(path):
            if path.exists():
                existing = pq.read_table(path, schema=_BAR_SCHEMA)
                existing_ids = set(existing.column("bar_id").to_pylist())
                new_ids = new_table.column("bar_id").to_pylist()
                if all(bid in existing_ids for bid in new_ids):
                    # Every new row's bar_id is already present; nothing to add.
                    return
                merged = pa.concat_tables([existing, new_table])
            else:
                merged = new_table
            merged = _dedup_by_timestamp(merged)
            _atomic_write_table(merged, path)

    def _sync_write_partition(
        self,
        instrument_id: uuid.UUID,
        year: int,
        bar_seconds: int,
        partition_bars: list[MarketBar],
    ) -> None:
        """Blocking read-merge-write for one ``(inst, year, bucket)`` partition.

        Kept synchronous so completion is deterministic across local WSL and
        CI filesystems. The bucket is derived from ``bar_seconds`` -- the
        caller has already grouped bars so every entry in
        ``partition_bars`` shares the same bucket (daily vs intraday).
        """
        # Lazy migration: absorb any pre-split legacy file for this
        # (inst, year) before we touch the new-layout file. This is a no-op
        # once the legacy file is gone, so the cost is one stat() per
        # partition write after the first.
        self._migrate_legacy_partition_if_present(instrument_id, year)

        path = self._bar_path(instrument_id, year, bar_seconds)
        with _exclusive_file_lock(path):
            existing_ids: set[str] = set()
            existing_table: pa.Table | None = None

            if path.exists():
                existing_table = pq.read_table(path, schema=_BAR_SCHEMA)
                existing_ids = set(existing_table.column("bar_id").to_pylist())

            new_rows = [b for b in partition_bars if str(b.bar_id) not in existing_ids]
            if not new_rows:
                return

            arrays = {
                "bar_id": [str(b.bar_id) for b in new_rows],
                "instrument_id": [str(b.instrument_id) for b in new_rows],
                "timestamp": [b.timestamp for b in new_rows],
                "bar_seconds": [b.bar_seconds for b in new_rows],
                "open": [float(b.open) for b in new_rows],
                "high": [float(b.high) for b in new_rows],
                "low": [float(b.low) for b in new_rows],
                "close": [float(b.close) for b in new_rows],
                "volume": [b.volume for b in new_rows],
                "vwap": [float(b.vwap) if b.vwap else None for b in new_rows],
                "is_complete": [b.is_complete for b in new_rows],
            }
            new_table = pa.table(arrays, schema=_BAR_SCHEMA)

            merged = (
                pa.concat_tables([existing_table, new_table])
                if existing_table is not None
                else new_table
            )

            # Dedup-on-write: when two ingest sources (e.g. IB + Tiingo) write
            # the same (instrument, timestamp, bar_seconds) with different
            # bar_ids the bar_id pre-filter above does not catch it, so the
            # parquet ends up with multiple rows per day. Downstream feature
            # builders then compute pct_change across the duplicates and
            # produce 1700%+ "single-day" returns, tripping the vol_forecast
            # bounds. Keep the last (most recently ingested) row per
            # (timestamp, bar_seconds).
            merged = _dedup_by_timestamp(merged)

            _atomic_write_table(merged, path)
        log.debug(
            "parquet_bar_store.wrote",
            instrument_id=str(instrument_id),
            year=year,
            bucket=bucket_for(bar_seconds),
            new_bars=len(new_rows),
        )

    async def store_bars(self, bars: list[MarketBar]) -> None:
        if not bars:
            return

        # Partition by (instrument, year, bucket): daily and intraday rows
        # for the same (inst, year) go to two separate files now.
        by_partition: dict[tuple[uuid.UUID, int, str], list[MarketBar]] = {}
        for bar in bars:
            key = (bar.instrument_id, bar.timestamp.year, bucket_for(bar.bar_seconds))
            by_partition.setdefault(key, []).append(bar)

        # Offload the blocking parquet IO to a worker thread so this
        # coroutine does not pin the event loop (broker pump and lock-lease
        # renewal coroutines must keep running during long writes).
        for (instrument_id, year, _bucket), partition_bars in by_partition.items():
            # Any bar in the group can stand in for the bucket -- they all
            # share one.
            bar_seconds = partition_bars[0].bar_seconds
            await asyncio.to_thread(
                self._sync_write_partition,
                instrument_id,
                year,
                bar_seconds,
                partition_bars,
            )

    async def get_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        years = range(start.year, end.year + 1)
        bars: list[MarketBar] = []

        for year in years:
            # New-layout bucketed file.
            new_path = self._bar_path(instrument_id, year, bar_seconds)
            if new_path.exists():
                table = await asyncio.to_thread(pq.read_table, new_path, schema=_BAR_SCHEMA)
                bars.extend(_rows_to_market_bars(table, bar_seconds, start, end))

            # Legacy mixed file for the same (inst, year) may still exist
            # during the transition window. Read it too and filter on
            # bar_seconds so a daily query never picks up intraday rows.
            legacy_path = self._legacy_bar_path(instrument_id, year)
            if legacy_path.exists():
                table = await asyncio.to_thread(pq.read_table, legacy_path, schema=_BAR_SCHEMA)
                bars.extend(_rows_to_market_bars(table, bar_seconds, start, end))

        actions = await self.get_corporate_actions(instrument_id, start.date())
        if actions:
            bars = apply_adjustments(bars, actions)

        bars.sort(key=lambda b: b.timestamp)
        return bars


def _rows_to_market_bars(
    table: pa.Table,
    bar_seconds: int,
    start: datetime,
    end: datetime,
) -> list[MarketBar]:
    """Materialise a parquet table into MarketBar objects with frequency + window filter."""
    out: list[MarketBar] = []
    for row in table.to_pylist():
        if row["bar_seconds"] != bar_seconds:
            continue
        ts = row["timestamp"]
        if ts < start or ts > end:
            continue
        out.append(
            MarketBar(
                bar_id=uuid.UUID(row["bar_id"]),
                instrument_id=uuid.UUID(row["instrument_id"]),
                timestamp=ts,
                bar_seconds=row["bar_seconds"],
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=row["volume"],
                vwap=Decimal(str(row["vwap"])) if row["vwap"] is not None else None,
                is_complete=row["is_complete"],
            )
        )
    return out


def _dedup_by_timestamp(table: pa.Table) -> pa.Table:
    """Keep the last row per (timestamp, bar_seconds).

    Two ingest sources writing the same (instrument, day, granularity) with
    different bar_ids would otherwise both land in the partition.
    """
    if table.num_rows <= 1:
        return table
    seen: dict[tuple[object, object], int] = {}
    timestamps = table.column("timestamp").to_pylist()
    seconds = table.column("bar_seconds").to_pylist()
    for index, key in enumerate(zip(timestamps, seconds, strict=True)):
        seen[key] = index
    if len(seen) == table.num_rows:
        return table
    return table.take(sorted(seen.values()))
