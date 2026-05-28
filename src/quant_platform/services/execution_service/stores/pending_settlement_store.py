"""Durable pending-settlement and completed-order state.

Before this module the ``AccountStateCoordinator`` kept
``_pending_lots`` and ``_completed_order_ids`` in process memory.  A
mid-day restart lost the settlement projection queue until the next
broker sync and created a window where a fill already applied pre-
restart could be re-processed.  The two stores here persist that state
in Postgres (Alembic revision ``005``) so a fresh coordinator rehydrates
the same view the previous process had.

Two stores intentionally stay separate:

- :class:`PendingSettlementStore` owns the sell-side lots waiting for
  T+1 / T+2 maturity.  ``advance_settlements`` reads lots whose
  ``settlement_date <= today`` and deletes them on success.
- :class:`CompletedOrderHintStore` remembers which order ids already saw
  their terminal ``BrokerOrderCompleted`` signal so a restart does not
  re-credit the reservation.

Each store follows the in-memory / Postgres split used by
:mod:`kill_switch_store`; bootstrap composition selects the right backend
based on the configured Postgres DSN.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

import structlog

from quant_platform.core.domain.settlement import SettlementLot, SettlementStatus
from quant_platform.services.execution_service.stores.completed_order_hint_store import (
    CompletedOrderHintStore,
    InMemoryCompletedOrderHintStore,
    PostgresCompletedOrderHintStore,
)

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

__all__ = [
    "CompletedOrderHintStore",
    "HydratedState",
    "InMemoryCompletedOrderHintStore",
    "InMemoryPendingSettlementStore",
    "PendingSettlementStore",
    "PostgresCompletedOrderHintStore",
    "PostgresPendingSettlementStore",
    "hydrate_account_state",
]


# ---------------------------------------------------------------------------
# Pending settlement lots
# ---------------------------------------------------------------------------


class PendingSettlementStore(Protocol):
    """Protocol for reading / writing durable pending settlement lots."""

    async def list_all(self, run_id: uuid.UUID | None = None) -> list[SettlementLot]: ...

    async def upsert(
        self, lot: SettlementLot, *, run_id: uuid.UUID, order_id: uuid.UUID
    ) -> None: ...

    async def delete(self, lot_id: uuid.UUID) -> None: ...


class InMemoryPendingSettlementStore:
    """Process-local store used when no Postgres DSN is configured."""

    def __init__(self) -> None:
        self._rows: dict[uuid.UUID, tuple[SettlementLot, uuid.UUID | None]] = {}

    async def list_all(self, run_id: uuid.UUID | None = None) -> list[SettlementLot]:
        if run_id is None:
            return [lot for lot, _ in self._rows.values()]
        return [lot for lot, stored_run in self._rows.values() if stored_run == run_id]

    async def upsert(self, lot: SettlementLot, *, run_id: uuid.UUID, order_id: uuid.UUID) -> None:
        self._rows[lot.lot_id] = (lot, run_id)

    async def delete(self, lot_id: uuid.UUID) -> None:
        self._rows.pop(lot_id, None)


class PostgresPendingSettlementStore:
    """Postgres-backed pending settlement lot store.

    Uses the ``pending_settlement_lots`` table from Alembic revision
    ``005``.  Columns are text / numeric / date — deliberately simple to
    avoid coupling the infra layer to the richer domain types.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def list_all(self, run_id: uuid.UUID | None = None) -> list[SettlementLot]:
        from sqlalchemy import text

        query = (
            "SELECT lot_id, fill_id, order_id, instrument_id, trade_date, "
            "settlement_date, gross_proceeds, commission, net_proceeds, "
            "currency FROM pending_settlement_lots"
        )
        params: dict[str, object] = {}
        if run_id is not None:
            query += " WHERE run_id = :run_id"
            params["run_id"] = str(run_id)

        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(query), params)).mappings().all()

        out: list[SettlementLot] = []
        for row in rows:
            try:
                out.append(
                    SettlementLot(
                        lot_id=uuid.UUID(str(row["lot_id"])),
                        fill_id=uuid.UUID(str(row["fill_id"])),
                        instrument_id=uuid.UUID(str(row["instrument_id"])),
                        trade_date=row["trade_date"],
                        settlement_date=row["settlement_date"],
                        gross_proceeds=Decimal(str(row["gross_proceeds"])),
                        commission=Decimal(str(row["commission"])),
                        net_proceeds=Decimal(str(row["net_proceeds"])),
                        currency=row["currency"],
                        status=SettlementStatus.PENDING,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.error(
                    "pending_settlement_store.row_decode_failed",
                    lot_id=row.get("lot_id"),
                    error=str(exc),
                )
        return out

    async def upsert(self, lot: SettlementLot, *, run_id: uuid.UUID, order_id: uuid.UUID) -> None:
        from sqlalchemy import text

        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO pending_settlement_lots (
                        lot_id, fill_id, order_id, instrument_id, trade_date,
                        settlement_date, gross_proceeds, commission,
                        net_proceeds, currency, run_id, created_at
                    )
                    VALUES (
                        :lot_id, :fill_id, :order_id, :instrument_id, :trade_date,
                        :settlement_date, :gross_proceeds, :commission,
                        :net_proceeds, :currency, :run_id, :created_at
                    )
                    ON CONFLICT (lot_id) DO UPDATE SET
                        gross_proceeds = EXCLUDED.gross_proceeds,
                        commission = EXCLUDED.commission,
                        net_proceeds = EXCLUDED.net_proceeds,
                        settlement_date = EXCLUDED.settlement_date
                    """
                ),
                {
                    "lot_id": str(lot.lot_id),
                    "fill_id": str(lot.fill_id),
                    "order_id": str(order_id),
                    "instrument_id": str(lot.instrument_id),
                    "trade_date": lot.trade_date,
                    "settlement_date": lot.settlement_date,
                    "gross_proceeds": lot.gross_proceeds,
                    "commission": lot.commission,
                    "net_proceeds": lot.net_proceeds,
                    "currency": lot.currency,
                    "run_id": str(run_id),
                    "created_at": datetime.now(tz=UTC),
                },
            )

    async def delete(self, lot_id: uuid.UUID) -> None:
        from sqlalchemy import text

        async with self._engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM pending_settlement_lots WHERE lot_id = :lot_id"),
                {"lot_id": str(lot_id)},
            )


@dataclass(frozen=True)
class HydratedState:
    pending_lots: list[SettlementLot]
    completed_order_ids: set[uuid.UUID]


async def hydrate_account_state(
    *,
    pending_store: PendingSettlementStore,
    completed_store: CompletedOrderHintStore,
    run_id: uuid.UUID,
) -> HydratedState:
    """Return the durable pending/completed state scoped to ``run_id``."""
    lots = await pending_store.list_all(run_id=run_id)
    completed = await completed_store.list_all(run_id=run_id)
    return HydratedState(pending_lots=lots, completed_order_ids=completed)
