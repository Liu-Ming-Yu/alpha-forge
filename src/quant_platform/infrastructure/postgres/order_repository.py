"""PostgreSQL order repository adapter."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import text

from quant_platform.core.domain.orders import (
    FillEvent,
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.infrastructure.postgres.row_coercion import require_datetime
from quant_platform.infrastructure.postgres.support import retry_transient

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.engine import RowMapping
    from sqlalchemy.ext.asyncio import AsyncEngine

_log = structlog.get_logger(__name__)


class PostgresOrderRepository:
    """PostgreSQL-backed OrderRepository."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @retry_transient()
    async def save_intent(self, intent: OrderIntent) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO order_intents
                        (order_id, strategy_run_id, portfolio_target_id,
                         instrument_id, side, quantity, order_type,
                         time_in_force, created_at, limit_price,
                         cash_reservation_id)
                    VALUES
                        (:order_id, :strategy_run_id, :portfolio_target_id,
                         :instrument_id, :side, :quantity, :order_type,
                         :time_in_force, :created_at, :limit_price,
                         :cash_reservation_id)
                """),
                {
                    "order_id": intent.order_id,
                    "strategy_run_id": intent.strategy_run_id,
                    "portfolio_target_id": intent.portfolio_target_id,
                    "instrument_id": intent.instrument_id,
                    "side": intent.side.value,
                    "quantity": intent.quantity,
                    "order_type": intent.order_type.value,
                    "time_in_force": intent.time_in_force.value,
                    "created_at": intent.created_at,
                    "limit_price": str(intent.limit_price) if intent.limit_price else None,
                    "cash_reservation_id": intent.cash_reservation_id,
                },
            )

    @retry_transient()
    async def get_intent(self, order_id: uuid.UUID) -> OrderIntent | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("SELECT * FROM order_intents WHERE order_id = :oid"),
                        {"oid": order_id},
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return _row_to_intent(row)

    @retry_transient()
    async def save_fill(self, fill: FillEvent) -> None:
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text("""
                    INSERT INTO fill_events
                        (fill_id, order_id, broker_order_id, broker_execution_id, instrument_id,
                         side, quantity, fill_price, commission, currency,
                         executed_at, received_at, supersedes_id)
                    VALUES
                        (:fill_id, :order_id, :broker_order_id, :broker_execution_id,
                         :instrument_id,
                         :side, :quantity, :fill_price, :commission, :currency,
                         :executed_at, :received_at, :supersedes_id)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "fill_id": fill.fill_id,
                    "order_id": fill.order_id,
                    "broker_order_id": fill.broker_order_id,
                    "broker_execution_id": fill.broker_execution_id,
                    "instrument_id": fill.instrument_id,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "fill_price": str(fill.fill_price),
                    "commission": str(fill.commission),
                    "currency": fill.currency,
                    "executed_at": fill.executed_at,
                    "received_at": fill.received_at,
                    "supersedes_id": fill.supersedes_id,
                },
            )
            # ON CONFLICT DO NOTHING silently absorbs duplicate (broker_order_id,
            # broker_execution_id) inserts — emit a structured log so the audit
            # trail still records that a duplicate fill was received.
            if result.rowcount == 0:
                _log.info(
                    "postgres_order_repository.fill_dedup_dropped",
                    fill_id=str(fill.fill_id),
                    order_id=str(fill.order_id),
                    broker_order_id=fill.broker_order_id,
                    broker_execution_id=fill.broker_execution_id,
                )

    @retry_transient()
    async def record_fill_slippage(
        self,
        fill_id: uuid.UUID,
        expected_price: Decimal,
        fill_price: Decimal,
    ) -> None:
        """Write slippage_bps for a previously saved fill."""
        if expected_price <= Decimal("0"):
            return
        # Keep slippage Decimal end-to-end so precision survives the round-trip
        # to the NUMERIC ``slippage_bps`` column. The prior float() cast lost
        # ~5 sig-figs which compound across many fills.
        slippage_bps = abs(fill_price - expected_price) / expected_price * Decimal("10000")
        async with self._engine.begin() as conn:
            await conn.execute(
                text("UPDATE fill_events SET slippage_bps = :bps WHERE fill_id = :fid"),
                {"bps": slippage_bps, "fid": fill_id},
            )

    @retry_transient()
    async def get_fills(self, order_id: uuid.UUID) -> list[FillEvent]:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text(
                            "SELECT * FROM fill_events WHERE order_id = :oid ORDER BY executed_at"
                        ),
                        {"oid": order_id},
                    )
                )
                .mappings()
                .all()
            )
        return [_row_to_fill(r) for r in rows]

    @retry_transient()
    async def list_open_orders(self, strategy_run_id: uuid.UUID) -> list[OrderIntent]:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text(
                            "SELECT * FROM order_intents "
                            "WHERE strategy_run_id = :srid AND is_terminal = FALSE"
                        ),
                        {"srid": strategy_run_id},
                    )
                )
                .mappings()
                .all()
            )
        return [_row_to_intent(r) for r in rows]

    @retry_transient()
    async def mark_terminal(self, order_id: uuid.UUID, reason: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE order_intents SET is_terminal = TRUE, "
                    "terminal_reason = :reason WHERE order_id = :oid"
                ),
                {"oid": order_id, "reason": reason},
            )


def _row_to_intent(row: Mapping[str, Any] | RowMapping) -> OrderIntent:
    lp = row["limit_price"]
    return OrderIntent(
        order_id=uuid.UUID(str(row["order_id"])),
        strategy_run_id=uuid.UUID(str(row["strategy_run_id"])),
        portfolio_target_id=uuid.UUID(str(row["portfolio_target_id"])),
        instrument_id=uuid.UUID(str(row["instrument_id"])),
        side=OrderSide(str(row["side"])),
        quantity=int(str(row["quantity"])),
        order_type=OrderType(str(row["order_type"])),
        time_in_force=TimeInForce(str(row["time_in_force"])),
        created_at=require_datetime(row, "created_at"),
        limit_price=Decimal(str(lp)) if lp is not None else None,
        cash_reservation_id=(
            uuid.UUID(str(row["cash_reservation_id"])) if row["cash_reservation_id"] else None
        ),
    )


def _row_to_fill(row: Mapping[str, Any] | RowMapping) -> FillEvent:
    return FillEvent(
        fill_id=uuid.UUID(str(row["fill_id"])),
        order_id=uuid.UUID(str(row["order_id"])),
        broker_order_id=str(row["broker_order_id"]),
        broker_execution_id=(
            str(row["broker_execution_id"]) if row.get("broker_execution_id") else None
        ),
        instrument_id=uuid.UUID(str(row["instrument_id"])),
        side=OrderSide(str(row["side"])),
        quantity=int(str(row["quantity"])),
        fill_price=Decimal(str(row["fill_price"])),
        commission=Decimal(str(row["commission"])),
        currency=str(row["currency"]),
        executed_at=require_datetime(row, "executed_at"),
        received_at=require_datetime(row, "received_at"),
        supersedes_id=(uuid.UUID(str(row["supersedes_id"])) if row["supersedes_id"] else None),
    )
