"""Postgres-backed V2 order-state and execution-quality repositories."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.v2.postgres_mappers import _row_to_order_state_event

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.domain.orders import ExecutionQualityReport, OrderStateEvent


class PostgresOrderStateStore:
    """Postgres-backed append-only OMS event store."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def append(self, event: OrderStateEvent) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO order_state_events
                        (event_id, order_id, event_type, occurred_at, status,
                         broker_order_id, idempotency_key, payload_json)
                    VALUES
                        (:event_id, :order_id, :event_type, :occurred_at, :status,
                         :broker_order_id, :idempotency_key, CAST(:payload_json AS JSONB))
                    ON CONFLICT (idempotency_key) DO NOTHING
                """),
                {
                    "event_id": event.event_id,
                    "order_id": event.order_id,
                    "event_type": event.event_type.value,
                    "occurred_at": event.occurred_at,
                    "status": event.status.value if event.status else None,
                    "broker_order_id": event.broker_order_id,
                    "idempotency_key": event.idempotency_key,
                    "payload_json": json.dumps(event.payload, default=str),
                },
            )

    async def list_events(self, order_id: uuid.UUID) -> list[OrderStateEvent]:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT *
                            FROM order_state_events
                            WHERE order_id = :order_id
                            ORDER BY occurred_at, event_id
                        """),
                        {"order_id": order_id},
                    )
                )
                .mappings()
                .all()
            )
        return [_row_to_order_state_event(row) for row in rows]

    async def latest(self, order_id: uuid.UUID) -> OrderStateEvent | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT *
                            FROM order_state_events
                            WHERE order_id = :order_id
                            ORDER BY occurred_at DESC, event_id DESC
                            LIMIT 1
                        """),
                        {"order_id": order_id},
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_order_state_event(row) if row else None


class PostgresExecutionQualityRepository:
    """Postgres-backed execution quality report repository."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save_execution_quality(self, report: ExecutionQualityReport) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO execution_quality_reports
                        (report_id, order_id, as_of, venue, tactic, arrival_price,
                         decision_price, vwap, fill_price, slippage_bps,
                         participation_rate, passed)
                    VALUES
                        (:report_id, :order_id, :as_of, :venue, :tactic, :arrival_price,
                         :decision_price, :vwap, :fill_price, :slippage_bps,
                         :participation_rate, :passed)
                    ON CONFLICT (report_id) DO NOTHING
                """),
                {
                    "report_id": report.report_id,
                    "order_id": report.order_id,
                    "as_of": report.as_of,
                    "venue": report.venue,
                    "tactic": report.tactic.value,
                    "arrival_price": report.arrival_price,
                    "decision_price": report.decision_price,
                    "vwap": report.vwap,
                    "fill_price": report.fill_price,
                    "slippage_bps": report.slippage_bps,
                    "participation_rate": report.participation_rate,
                    "passed": report.passed,
                },
            )
