"""Corporate-action methods for the Parquet bar store."""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from quant_platform.core.domain.instruments import CorporateAction, CorporateActionType
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.reference.corporate_actions import apply_adjustments
from quant_platform.services.data_service.stores.parquet_store_io import (
    BAR_SCHEMA,
    CA_SCHEMA,
    atomic_write_table,
    exclusive_file_lock,
)

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger(__name__)


class ParquetCorporateActionsMixin:
    """Corporate-action persistence and adjusted-partition reprocessing."""

    _root: Path

    def _ca_path(self, instrument_id: uuid.UUID) -> Path:
        raise NotImplementedError

    def _adjusted_bar_path(self, instrument_id: uuid.UUID, year: int) -> Path:
        raise NotImplementedError

    async def get_corporate_actions(
        self,
        instrument_id: uuid.UUID,
        since: date,
    ) -> list[CorporateAction]:
        """Return all known corporate actions for ``instrument_id``."""
        del since  # read-time adjustment filters per bar.
        path = self._ca_path(instrument_id)
        if not path.exists():
            return []

        # Read without enforcing the current schema: pre-existing files may
        # carry ``float64`` ratio/cash_amount columns. The
        # ``Decimal(str(value))`` reconstruction below works for both float
        # and string column types. Offloaded to a worker thread to avoid
        # blocking the event loop on disk IO.
        table = await asyncio.to_thread(pq.read_table, path)
        result: list[CorporateAction] = []
        for row in table.to_pylist():
            result.append(
                CorporateAction(
                    action_id=uuid.UUID(row["action_id"]),
                    instrument_id=uuid.UUID(row["instrument_id"]),
                    action_type=CorporateActionType(row["action_type"]),
                    ex_date=row["ex_date"],
                    record_date=row["record_date"],
                    pay_date=row["pay_date"],
                    ratio=Decimal(str(row["ratio"])),
                    cash_amount=Decimal(str(row["cash_amount"])),
                    currency=row["currency"],
                    supersedes_id=uuid.UUID(row["supersedes_id"]) if row["supersedes_id"] else None,
                    notes=row["notes"],
                )
            )
        return result

    async def store_corporate_action(self, action: CorporateAction) -> None:
        path = self._ca_path(action.instrument_id)
        await asyncio.to_thread(self._sync_store_corporate_action, action, path)

    def _sync_store_corporate_action(
        self,
        action: CorporateAction,
        path: Path,
    ) -> None:
        with exclusive_file_lock(path):
            existing: pa.Table | None = None
            if path.exists():
                existing = _normalise_ca_table(pq.read_table(path))
                existing_ids = set(existing.column("action_id").to_pylist())
                if str(action.action_id) in existing_ids:
                    return

            row = {
                "action_id": [str(action.action_id)],
                "instrument_id": [str(action.instrument_id)],
                "action_type": [action.action_type.value],
                "ex_date": [action.ex_date],
                "record_date": [action.record_date],
                "pay_date": [action.pay_date],
                # Persist Decimals as their canonical string representation
                # so the value round-trips losslessly (float64 truncates).
                "ratio": [str(action.ratio)],
                "cash_amount": [str(action.cash_amount)],
                "currency": [action.currency],
                "supersedes_id": [str(action.supersedes_id) if action.supersedes_id else ""],
                "notes": [action.notes],
            }
            new_table = pa.table(row, schema=CA_SCHEMA)
            merged = pa.concat_tables([existing, new_table]) if existing is not None else new_table
            atomic_write_table(merged, path)

    async def reprocess_corporate_actions(
        self,
        instrument_id: uuid.UUID,
    ) -> int:
        """Re-emit adjusted partitions for ``instrument_id`` after a late CA."""
        raw_dir = self._root / "bars" / str(instrument_id)
        if not raw_dir.exists():
            log.info(
                "parquet_bar_store.reprocess_ca.no_bars",
                instrument_id=str(instrument_id),
            )
            return 0

        actions = await self.get_corporate_actions(
            instrument_id,
            since=date.min,
        )

        # Walk the post-split bucketed subdirectories AND the legacy
        # top-level files. A given (instrument, year) may have rows in any
        # combination of bars/<inst>/daily/<year>.parquet,
        # bars/<inst>/intraday/<year>.parquet, and the legacy
        # bars/<inst>/<year>.parquet if it has not been migrated yet.
        files_by_year: dict[int, list[Path]] = {}
        for child in sorted(raw_dir.iterdir()):
            if child.is_dir() and child.name in {"daily", "intraday"}:
                for year_file in sorted(child.glob("*.parquet")):
                    try:
                        year = int(year_file.stem)
                    except ValueError:
                        continue
                    files_by_year.setdefault(year, []).append(year_file)
            elif child.is_file() and child.suffix == ".parquet":
                try:
                    year = int(child.stem)
                except ValueError:
                    continue
                files_by_year.setdefault(year, []).append(child)

        total = 0
        for year in sorted(files_by_year):
            raw_bars: list[MarketBar] = []
            for year_file in files_by_year[year]:
                table = await asyncio.to_thread(pq.read_table, year_file, schema=BAR_SCHEMA)
                raw_bars.extend(_market_bar_from_row(row) for row in table.to_pylist())

            adjusted = apply_adjustments(raw_bars, actions) if actions else raw_bars
            adjusted.sort(key=lambda bar: bar.timestamp)

            if not adjusted:
                continue

            out_path = self._adjusted_bar_path(instrument_id, year)
            await asyncio.to_thread(_sync_write_adjusted_partition, out_path, _bar_arrays(adjusted))
            total += len(adjusted)
            log.info(
                "parquet_bar_store.reprocess_ca.wrote",
                instrument_id=str(instrument_id),
                year=year,
                bars=len(adjusted),
            )

        log.info(
            "parquet_bar_store.reprocess_ca.complete",
            instrument_id=str(instrument_id),
            bars=total,
            actions=len(actions),
        )
        return total


def _market_bar_from_row(row: dict[str, object]) -> MarketBar:
    return MarketBar(
        bar_id=uuid.UUID(str(row["bar_id"])),
        instrument_id=uuid.UUID(str(row["instrument_id"])),
        timestamp=_datetime_value(row["timestamp"], "timestamp"),
        bar_seconds=_int_value(row["bar_seconds"], "bar_seconds"),
        open=Decimal(str(row["open"])),
        high=Decimal(str(row["high"])),
        low=Decimal(str(row["low"])),
        close=Decimal(str(row["close"])),
        volume=_int_value(row["volume"], "volume"),
        vwap=Decimal(str(row["vwap"])) if row["vwap"] is not None else None,
        is_complete=_bool_value(row["is_complete"], "is_complete"),
    )


def _datetime_value(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"{field_name} must be a datetime")


def _int_value(value: object, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be an integer")


def _bool_value(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a boolean")


def _sync_write_adjusted_partition(out_path: Path, arrays: dict[str, list[object]]) -> None:
    """Blocking helper: lock + atomic write for adjusted-partition output."""
    with exclusive_file_lock(out_path):
        atomic_write_table(pa.table(arrays, schema=BAR_SCHEMA), out_path)


def _normalise_ca_table(table: pa.Table) -> pa.Table:
    """Cast a float64-ratio CA table into the current string schema.

    Existing parquet files written with the prior float64 schema must be
    upgraded before they can be ``concat_tables``'d with a newly-written
    string-typed row.  pyarrow casts float64 → string column-wise; we
    preserve the canonical string form via Python ``str`` so a Decimal
    round-trip via ``Decimal(str(value))`` matches what the writer emits.
    """
    if table.schema.equals(CA_SCHEMA):
        return table

    columns = {name: table.column(name) for name in table.schema.names}
    for col_name in ("ratio", "cash_amount"):
        col = columns.get(col_name)
        if col is None:
            continue
        if pa.types.is_floating(col.type) or col.type != pa.string():
            columns[col_name] = pa.array([str(v) for v in col.to_pylist()], type=pa.string())
    return pa.table(columns, schema=CA_SCHEMA)


def _bar_arrays(bars: list[MarketBar]) -> dict[str, list[object]]:
    return {
        "bar_id": [str(bar.bar_id) for bar in bars],
        "instrument_id": [str(bar.instrument_id) for bar in bars],
        "timestamp": [bar.timestamp for bar in bars],
        "bar_seconds": [bar.bar_seconds for bar in bars],
        "open": [float(bar.open) for bar in bars],
        "high": [float(bar.high) for bar in bars],
        "low": [float(bar.low) for bar in bars],
        "close": [float(bar.close) for bar in bars],
        "volume": [bar.volume for bar in bars],
        "vwap": [float(bar.vwap) if bar.vwap else None for bar in bars],
        "is_complete": [bar.is_complete for bar in bars],
    }
