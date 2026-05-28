"""In-memory V2 instrument repository."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from quant_platform.core.domain.instruments import (
    SecurityMasterQuality,
    SecurityMasterRecord,
    SymbolHistory,
    UniverseSnapshot,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


class InMemoryInstrumentRepository:
    """Point-in-time security-master repository."""

    def __init__(self) -> None:
        self._records: dict[uuid.UUID, list[SecurityMasterRecord]] = defaultdict(list)
        self._symbol_history: list[SymbolHistory] = []
        self._universes: dict[str, list[UniverseSnapshot]] = defaultdict(list)

    async def upsert_security_master_record(self, record: SecurityMasterRecord) -> None:
        rows = self._records[record.instrument.instrument_id]
        rows[:] = [existing for existing in rows if existing.record_id != record.record_id]
        rows.append(record)
        rows.sort(key=lambda item: (item.available_at, item.as_of))

    async def get_record(
        self,
        instrument_id: uuid.UUID,
        *,
        as_of: datetime,
    ) -> SecurityMasterRecord | None:
        candidates = [
            row
            for row in self._records.get(instrument_id, [])
            if row.as_of <= as_of
            and row.available_at <= as_of
            and row.quality == SecurityMasterQuality.APPROVED
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: (row.available_at, row.as_of))

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
        self._symbol_history = [
            row for row in self._symbol_history if row.history_id != history.history_id
        ]
        self._symbol_history.append(history)

    async def resolve_symbol(self, symbol: str, *, as_of: datetime) -> uuid.UUID | None:
        symbol_date = as_of.date()
        candidates = [
            row
            for row in self._symbol_history
            if row.symbol == symbol
            and row.valid_from <= symbol_date
            and (row.valid_to is None or symbol_date <= row.valid_to)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: row.valid_from).instrument_id

    async def save_universe_snapshot(self, snapshot: UniverseSnapshot) -> None:
        rows = self._universes[snapshot.universe_name]
        rows[:] = [existing for existing in rows if existing.snapshot_id != snapshot.snapshot_id]
        rows.append(snapshot)
        rows.sort(key=lambda item: (item.available_at, item.as_of))

    async def latest_universe_snapshot(
        self,
        universe_name: str,
        *,
        as_of: datetime,
    ) -> UniverseSnapshot | None:
        candidates = [
            row
            for row in self._universes.get(universe_name, [])
            if row.as_of <= as_of
            and row.available_at <= as_of
            and row.quality == SecurityMasterQuality.APPROVED
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: (row.available_at, row.as_of))
