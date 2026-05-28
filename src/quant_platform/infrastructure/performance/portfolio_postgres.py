"""Portfolio-performance operations for the Postgres performance adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.performance.mappers import (
    row_to_instrument_pnl as _row_to_instrument_pnl,
)
from quant_platform.infrastructure.performance.mappers import row_to_nav as _row_to_nav
from quant_platform.infrastructure.performance.status import (
    build_performance_report as _build_performance_report,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.domain.production import (
        InstrumentPnl,
        NavSnapshot,
        PerformanceReport,
    )


class PostgresPortfolioPerformanceMixin:
    """NAV, instrument PnL, and slippage operations."""

    _engine: AsyncEngine

    async def save_nav_snapshot(self, snapshot: NavSnapshot) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO nav_snapshots
                        (snapshot_id, strategy_run_id, as_of, net_asset_value,
                         gross_exposure, cash, source, realized_pnl, unrealized_pnl)
                    VALUES
                        (:snapshot_id, :strategy_run_id, :as_of, :net_asset_value,
                         :gross_exposure, :cash, :source, :realized_pnl, :unrealized_pnl)
                    ON CONFLICT (snapshot_id) DO NOTHING
                """),
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "strategy_run_id": snapshot.strategy_run_id,
                    "as_of": snapshot.as_of,
                    "net_asset_value": snapshot.net_asset_value,
                    "gross_exposure": snapshot.gross_exposure,
                    "cash": snapshot.cash,
                    "source": snapshot.source,
                    "realized_pnl": snapshot.realized_pnl,
                    "unrealized_pnl": snapshot.unrealized_pnl,
                },
            )

    async def list_nav_snapshots(
        self,
        strategy_run_id: uuid.UUID,
        *,
        limit: int = 252,
    ) -> list[NavSnapshot]:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT snapshot_id, strategy_run_id, as_of, net_asset_value,
                                   gross_exposure, cash, source,
                                   realized_pnl, unrealized_pnl
                            FROM nav_snapshots
                            WHERE strategy_run_id = :strategy_run_id
                            ORDER BY as_of DESC
                            LIMIT :limit
                        """),
                        {"strategy_run_id": strategy_run_id, "limit": max(1, limit)},
                    )
                )
                .mappings()
                .all()
            )
        snapshots = [_row_to_nav(row) for row in rows]
        snapshots.sort(key=lambda row: row.as_of)
        return snapshots

    async def performance_report(
        self,
        strategy_run_id: uuid.UUID,
        *,
        as_of: datetime,
        window: int = 90,
    ) -> PerformanceReport:
        rows = await self.list_nav_snapshots(strategy_run_id, limit=window + 1)
        return _build_performance_report(strategy_run_id, as_of=as_of, rows=rows)

    async def record_instrument_pnl(self, record: InstrumentPnl) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO instrument_pnl
                        (pnl_id, strategy_run_id, instrument_id, as_of,
                         realized_pnl, unrealized_pnl, weight, contribution)
                    VALUES
                        (:pnl_id, :strategy_run_id, :instrument_id, :as_of,
                         :realized_pnl, :unrealized_pnl, :weight, :contribution)
                    ON CONFLICT (pnl_id) DO NOTHING
                """),
                {
                    "pnl_id": record.pnl_id,
                    "strategy_run_id": record.strategy_run_id,
                    "instrument_id": record.instrument_id,
                    "as_of": record.as_of,
                    "realized_pnl": str(record.realized_pnl),
                    "unrealized_pnl": str(record.unrealized_pnl),
                    "weight": str(record.weight),
                    "contribution": str(record.contribution),
                },
            )

    async def list_instrument_pnl(
        self,
        strategy_run_id: uuid.UUID,
        *,
        limit: int = 252,
    ) -> list[InstrumentPnl]:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT pnl_id, strategy_run_id, instrument_id, as_of,
                                   realized_pnl, unrealized_pnl, weight, contribution
                            FROM instrument_pnl
                            WHERE strategy_run_id = :strategy_run_id
                            ORDER BY as_of DESC, instrument_id
                            LIMIT :limit
                        """),
                        {"strategy_run_id": strategy_run_id, "limit": max(1, limit)},
                    )
                )
                .mappings()
                .all()
            )
        return [_row_to_instrument_pnl(row) for row in rows]

    async def average_slippage_bps(
        self,
        strategy_run_id: uuid.UUID,
        *,
        limit: int = 252,
    ) -> float | None:
        """Return average slippage in basis points for recent fills, or None."""
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT AVG(f.slippage_bps) AS avg_slippage
                            FROM fill_events f
                            JOIN order_intents o ON o.order_id = f.order_id
                            WHERE o.strategy_run_id = :strategy_run_id
                              AND f.slippage_bps IS NOT NULL
                            ORDER BY f.executed_at DESC
                            LIMIT :limit
                        """),
                        {"strategy_run_id": strategy_run_id, "limit": limit},
                    )
                )
                .mappings()
                .first()
            )
        if row is None or row["avg_slippage"] is None:
            return None
        return float(row["avg_slippage"])
