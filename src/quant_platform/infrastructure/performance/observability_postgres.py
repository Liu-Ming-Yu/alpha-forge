"""PostgreSQL persistence for runtime and broker observability evidence."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.performance.mappers import (
    row_to_broker_health as _row_to_broker_health,
)
from quant_platform.infrastructure.performance.mappers import (
    row_to_broker_smoke as _row_to_broker_smoke,
)
from quant_platform.infrastructure.performance.mappers import (
    row_to_heartbeat as _row_to_heartbeat,
)
from quant_platform.infrastructure.performance.mappers import (
    row_to_paper_lifecycle as _row_to_paper_lifecycle,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.domain.production import (
        BrokerHealthObservation,
        BrokerSmokeObservation,
        PaperLifecycleObservation,
        RuntimeHeartbeat,
    )


class PostgresObservabilityPerformanceMixin:
    """Runtime heartbeat and broker readiness persistence methods."""

    _engine: AsyncEngine

    async def save_runtime_heartbeat(self, heartbeat: RuntimeHeartbeat) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO runtime_heartbeats
                        (component, as_of, status, detail)
                    VALUES (:component, :as_of, :status, :detail)
                    ON CONFLICT (component)
                    DO UPDATE SET
                        as_of = EXCLUDED.as_of,
                        status = EXCLUDED.status,
                        detail = EXCLUDED.detail,
                        updated_at = now()
                """),
                {
                    "component": heartbeat.component,
                    "as_of": heartbeat.as_of,
                    "status": heartbeat.status,
                    "detail": heartbeat.detail,
                },
            )

    async def latest_runtime_heartbeat(self, component: str) -> RuntimeHeartbeat | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT component, as_of, status, detail
                            FROM runtime_heartbeats
                            WHERE component = :component
                        """),
                        {"component": component},
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_heartbeat(row) if row else None

    async def save_broker_health(self, observation: BrokerHealthObservation) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO broker_health_observations
                        (observation_id, observed_at, status, latency_ms,
                         last_heartbeat_at, detail)
                    VALUES
                        (:observation_id, :observed_at, :status, :latency_ms,
                         :last_heartbeat_at, :detail)
                """),
                {
                    "observation_id": uuid.uuid4(),
                    "observed_at": observation.observed_at,
                    "status": observation.status,
                    "latency_ms": observation.latency_ms,
                    "last_heartbeat_at": observation.last_heartbeat_at,
                    "detail": observation.detail,
                },
            )

    async def latest_broker_health(self) -> BrokerHealthObservation | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT observed_at, status, latency_ms, last_heartbeat_at, detail
                            FROM broker_health_observations
                            ORDER BY observed_at DESC
                            LIMIT 1
                        """)
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_broker_health(row) if row else None

    async def save_broker_smoke(self, observation: BrokerSmokeObservation) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO broker_smoke_observations
                        (observation_id, observed_at, status, host, port, client_id,
                         latency_ms, account_status, positions_status, open_orders_status, detail)
                    VALUES
                        (:observation_id, :observed_at, :status, :host, :port, :client_id,
                         :latency_ms, :account_status, :positions_status, :open_orders_status,
                         :detail)
                """),
                {
                    "observation_id": uuid.uuid4(),
                    "observed_at": observation.observed_at,
                    "status": observation.status,
                    "host": observation.host,
                    "port": observation.port,
                    "client_id": observation.client_id,
                    "latency_ms": observation.latency_ms,
                    "account_status": observation.account_status,
                    "positions_status": observation.positions_status,
                    "open_orders_status": observation.open_orders_status,
                    "detail": observation.detail,
                },
            )

    async def latest_broker_smoke(self) -> BrokerSmokeObservation | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT observed_at, status, host, port, client_id,
                                   latency_ms, account_status, positions_status,
                                   open_orders_status, detail
                            FROM broker_smoke_observations
                            ORDER BY observed_at DESC
                            LIMIT 1
                        """)
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_broker_smoke(row) if row else None

    async def save_paper_lifecycle(self, observation: PaperLifecycleObservation) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO paper_lifecycle_observations
                        (observation_id, observed_at, status, host, port, client_id,
                         instrument_id, broker_order_id, max_notional_usd, limit_price,
                         quantity, ack_status, cancel_status, stale_open_order_count, detail)
                    VALUES
                        (:observation_id, :observed_at, :status, :host, :port, :client_id,
                         :instrument_id, :broker_order_id, :max_notional_usd, :limit_price,
                         :quantity, :ack_status, :cancel_status, :stale_open_order_count,
                         :detail)
                """),
                {
                    "observation_id": uuid.uuid4(),
                    "observed_at": observation.observed_at,
                    "status": observation.status,
                    "host": observation.host,
                    "port": observation.port,
                    "client_id": observation.client_id,
                    "instrument_id": observation.instrument_id,
                    "broker_order_id": observation.broker_order_id,
                    "max_notional_usd": observation.max_notional_usd,
                    "limit_price": observation.limit_price,
                    "quantity": observation.quantity,
                    "ack_status": observation.ack_status,
                    "cancel_status": observation.cancel_status,
                    "stale_open_order_count": observation.stale_open_order_count,
                    "detail": observation.detail,
                },
            )

    async def latest_paper_lifecycle(self) -> PaperLifecycleObservation | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT observed_at, status, host, port, client_id,
                                   instrument_id, broker_order_id, max_notional_usd,
                                   limit_price, quantity, ack_status, cancel_status,
                                   stale_open_order_count, detail
                            FROM paper_lifecycle_observations
                            ORDER BY observed_at DESC
                            LIMIT 1
                        """)
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_paper_lifecycle(row) if row else None
