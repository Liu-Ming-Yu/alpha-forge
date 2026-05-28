"""Postgres-backed V2 instrument master repository."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.v2.postgres_mappers import (
    _row_to_security_record,
    _row_to_universe_snapshot,
)

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.domain.instruments import (
        SecurityMasterRecord,
        SymbolHistory,
        UniverseSnapshot,
    )


class PostgresInstrumentRepository:
    """Postgres-backed point-in-time security master."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def upsert_security_master_record(self, record: SecurityMasterRecord) -> None:
        instrument = record.instrument
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO security_master_records
                        (record_id, instrument_id, symbol, exchange, asset_class,
                         currency, lot_size, active, sector, primary_exchange, country,
                         identifiers_json, as_of, available_at, source, quality_status)
                    VALUES
                        (:record_id, :instrument_id, :symbol, :exchange, :asset_class,
                         :currency, :lot_size, :active, :sector, :primary_exchange, :country,
                         CAST(:identifiers_json AS JSONB), :as_of, :available_at,
                         :source, :quality_status)
                    ON CONFLICT (record_id) DO UPDATE SET
                        symbol = EXCLUDED.symbol,
                        exchange = EXCLUDED.exchange,
                        asset_class = EXCLUDED.asset_class,
                        currency = EXCLUDED.currency,
                        lot_size = EXCLUDED.lot_size,
                        active = EXCLUDED.active,
                        sector = EXCLUDED.sector,
                        primary_exchange = EXCLUDED.primary_exchange,
                        country = EXCLUDED.country,
                        identifiers_json = EXCLUDED.identifiers_json,
                        as_of = EXCLUDED.as_of,
                        available_at = EXCLUDED.available_at,
                        source = EXCLUDED.source,
                        quality_status = EXCLUDED.quality_status
                """),
                {
                    "record_id": record.record_id,
                    "instrument_id": instrument.instrument_id,
                    "symbol": instrument.symbol,
                    "exchange": instrument.exchange,
                    "asset_class": instrument.asset_class.value,
                    "currency": instrument.currency,
                    "lot_size": instrument.lot_size,
                    "active": instrument.active,
                    "sector": instrument.sector,
                    "primary_exchange": record.primary_exchange,
                    "country": record.country,
                    "identifiers_json": json.dumps(record.identifiers, default=str),
                    "as_of": record.as_of,
                    "available_at": record.available_at,
                    "source": record.source,
                    "quality_status": record.quality.value,
                },
            )

    async def get_record(
        self,
        instrument_id: uuid.UUID,
        *,
        as_of: datetime,
    ) -> SecurityMasterRecord | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT *
                            FROM security_master_records
                            WHERE instrument_id = :instrument_id
                              AND as_of <= :as_of
                              AND available_at <= :as_of
                              AND quality_status = 'approved'
                            ORDER BY available_at DESC, as_of DESC
                            LIMIT 1
                        """),
                        {"instrument_id": instrument_id, "as_of": as_of},
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_security_record(row) if row else None

    async def require_record(
        self,
        instrument_id: uuid.UUID,
        *,
        as_of: datetime,
    ) -> SecurityMasterRecord:
        record = await self.get_record(instrument_id, as_of=as_of)
        if record is None:
            raise LookupError(f"missing approved security-master record for {instrument_id}")
        return record

    async def add_symbol_history(self, history: SymbolHistory) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO symbol_history
                        (history_id, instrument_id, symbol, valid_from, valid_to, source)
                    VALUES
                        (:history_id, :instrument_id, :symbol, :valid_from, :valid_to, :source)
                    ON CONFLICT (history_id) DO NOTHING
                """),
                {
                    "history_id": history.history_id,
                    "instrument_id": history.instrument_id,
                    "symbol": history.symbol,
                    "valid_from": history.valid_from,
                    "valid_to": history.valid_to,
                    "source": history.source,
                },
            )

    async def resolve_symbol(self, symbol: str, *, as_of: datetime) -> uuid.UUID | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT instrument_id
                            FROM symbol_history
                            WHERE symbol = :symbol
                              AND valid_from <= :as_of_date
                              AND (valid_to IS NULL OR valid_to >= :as_of_date)
                            ORDER BY valid_from DESC
                            LIMIT 1
                        """),
                        {"symbol": symbol, "as_of_date": as_of.date()},
                    )
                )
                .mappings()
                .first()
            )
        return uuid.UUID(str(row["instrument_id"])) if row else None

    async def save_universe_snapshot(self, snapshot: UniverseSnapshot) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO universe_snapshots
                        (snapshot_id, universe_name, as_of, available_at,
                         instrument_ids_json, source, quality_status)
                    VALUES
                        (:snapshot_id, :universe_name, :as_of, :available_at,
                         CAST(:instrument_ids_json AS JSONB), :source, :quality_status)
                    ON CONFLICT (snapshot_id) DO NOTHING
                """),
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "universe_name": snapshot.universe_name,
                    "as_of": snapshot.as_of,
                    "available_at": snapshot.available_at,
                    "instrument_ids_json": json.dumps([str(i) for i in snapshot.instrument_ids]),
                    "source": snapshot.source,
                    "quality_status": snapshot.quality.value,
                },
            )

    async def latest_universe_snapshot(
        self,
        universe_name: str,
        *,
        as_of: datetime,
    ) -> UniverseSnapshot | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT *
                            FROM universe_snapshots
                            WHERE universe_name = :universe_name
                              AND as_of <= :as_of
                              AND available_at <= :as_of
                              AND quality_status = 'approved'
                            ORDER BY available_at DESC, as_of DESC
                            LIMIT 1
                        """),
                        {"universe_name": universe_name, "as_of": as_of},
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_universe_snapshot(row) if row else None
