"""PostgreSQL position repository adapter."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.infrastructure.postgres.row_coercion import require_datetime
from quant_platform.infrastructure.postgres.support import retry_transient

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from sqlalchemy.engine import RowMapping
    from sqlalchemy.ext.asyncio import AsyncEngine


class PostgresPositionRepository:
    """PostgreSQL-backed PositionRepository."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @retry_transient()
    async def save_snapshot(self, snapshot: AccountSnapshot) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO account_snapshots
                        (snapshot_id, as_of, settled_cash, unsettled_cash,
                         reserved_cash, available_cash, net_asset_value, source)
                    VALUES
                        (:snapshot_id, :as_of, :settled_cash, :unsettled_cash,
                         :reserved_cash, :available_cash, :net_asset_value, :source)
                    ON CONFLICT (snapshot_id) DO NOTHING
                """),
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "as_of": snapshot.as_of,
                    "settled_cash": str(snapshot.settled_cash),
                    "unsettled_cash": str(snapshot.unsettled_cash),
                    "reserved_cash": str(snapshot.reserved_cash),
                    "available_cash": str(snapshot.available_cash),
                    "net_asset_value": str(snapshot.net_asset_value),
                    "source": snapshot.source,
                },
            )
            for pos in snapshot.positions:
                await conn.execute(
                    text("""
                        INSERT INTO position_snapshots
                            (snapshot_id, instrument_id, quantity, average_cost,
                             market_price, market_value, unrealised_pnl,
                             as_of, source)
                        VALUES
                            (:snapshot_id, :instrument_id, :quantity,
                             :average_cost, :market_price, :market_value,
                             :unrealised_pnl, :as_of, :source)
                        ON CONFLICT (snapshot_id, instrument_id) DO NOTHING
                    """),
                    {
                        "snapshot_id": snapshot.snapshot_id,
                        "instrument_id": pos.instrument_id,
                        "quantity": pos.quantity,
                        "average_cost": str(pos.average_cost),
                        "market_price": str(pos.market_price),
                        "market_value": str(pos.market_value),
                        "unrealised_pnl": str(pos.unrealised_pnl),
                        "as_of": pos.as_of,
                        "source": pos.source,
                    },
                )

    @retry_transient()
    async def get_latest_snapshot(self) -> AccountSnapshot | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("SELECT * FROM account_snapshots ORDER BY as_of DESC LIMIT 1")
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return await self._hydrate(row)

    @retry_transient()
    async def get_snapshot_at(self, as_of: datetime) -> AccountSnapshot | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text(
                            "SELECT * FROM account_snapshots "
                            "WHERE as_of <= :as_of ORDER BY as_of DESC LIMIT 1"
                        ),
                        {"as_of": as_of},
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return await self._hydrate(row)

    async def _hydrate(self, row: Mapping[str, Any] | RowMapping) -> AccountSnapshot:
        async with self._engine.connect() as conn:
            pos_rows = (
                (
                    await conn.execute(
                        text("SELECT * FROM position_snapshots WHERE snapshot_id = :sid"),
                        {"sid": row["snapshot_id"]},
                    )
                )
                .mappings()
                .all()
            )

        positions = tuple(
            PositionSnapshot(
                snapshot_id=uuid.UUID(str(p["snapshot_id"])),
                instrument_id=uuid.UUID(str(p["instrument_id"])),
                quantity=int(p["quantity"]),
                average_cost=Decimal(str(p["average_cost"])),
                market_price=Decimal(str(p["market_price"])),
                market_value=Decimal(str(p["market_value"])),
                unrealised_pnl=Decimal(str(p["unrealised_pnl"])),
                as_of=require_datetime(p, "as_of"),
                source=str(p["source"]),
            )
            for p in pos_rows
        )

        return AccountSnapshot(
            snapshot_id=uuid.UUID(str(row["snapshot_id"])),
            as_of=require_datetime(row, "as_of"),
            settled_cash=Decimal(str(row["settled_cash"])),
            unsettled_cash=Decimal(str(row["unsettled_cash"])),
            reserved_cash=Decimal(str(row["reserved_cash"])),
            available_cash=Decimal(str(row["available_cash"])),
            net_asset_value=Decimal(str(row["net_asset_value"])),
            positions=positions,
            source=str(row["source"]),
        )
