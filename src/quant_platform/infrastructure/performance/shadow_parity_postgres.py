"""PostgreSQL shadow-vs-paper parity operations for the performance repository."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.performance.mappers import (
    row_to_shadow_paper_parity as _row_to_shadow_paper_parity,
)
from quant_platform.infrastructure.performance.status import (
    build_shadow_paper_parity_status as _build_shadow_paper_parity_status,
)

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.domain.production import (
        ShadowPaperParityRecord,
        ShadowPaperParityStatus,
    )


class PostgresShadowPaperParityPerformanceMixin:
    """Shadow-vs-paper parity persistence methods."""

    _engine: AsyncEngine

    async def save_shadow_paper_parity(self, record: ShadowPaperParityRecord) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO shadow_paper_parity_observations
                        (parity_id, signal_name, signal_type, trading_day, as_of,
                         instruments_compared, missing_instruments,
                         max_target_weight_diff_bps, order_side_mismatches,
                         metadata_json)
                    VALUES
                        (:parity_id, :signal_name, :signal_type, :trading_day, :as_of,
                         :instruments_compared, :missing_instruments,
                         :max_target_weight_diff_bps, :order_side_mismatches,
                         CAST(:metadata AS JSONB))
                    ON CONFLICT (parity_id)
                    DO UPDATE SET
                        instruments_compared = EXCLUDED.instruments_compared,
                        missing_instruments = EXCLUDED.missing_instruments,
                        max_target_weight_diff_bps = EXCLUDED.max_target_weight_diff_bps,
                        order_side_mismatches = EXCLUDED.order_side_mismatches,
                        metadata_json = EXCLUDED.metadata_json
                """),
                {
                    "parity_id": record.parity_id,
                    "signal_name": record.signal_name,
                    "signal_type": record.signal_type,
                    "trading_day": record.trading_day,
                    "as_of": record.as_of,
                    "instruments_compared": record.instruments_compared,
                    "missing_instruments": record.missing_instruments,
                    "max_target_weight_diff_bps": record.max_target_weight_diff_bps,
                    "order_side_mismatches": record.order_side_mismatches,
                    "metadata": json.dumps(record.metadata, default=str),
                },
            )

    async def list_shadow_paper_parity(
        self,
        signal_name: str,
        signal_type: str,
        *,
        as_of: datetime,
        limit: int = 252,
    ) -> list[ShadowPaperParityRecord]:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT parity_id, signal_name, signal_type, trading_day, as_of,
                                   instruments_compared, missing_instruments,
                                   max_target_weight_diff_bps, order_side_mismatches,
                                   metadata_json
                            FROM shadow_paper_parity_observations
                            WHERE signal_name = :signal_name
                              AND signal_type = :signal_type
                              AND as_of <= :as_of
                            ORDER BY as_of DESC
                            LIMIT :limit
                        """),
                        {
                            "signal_name": signal_name,
                            "signal_type": signal_type,
                            "as_of": as_of,
                            "limit": max(1, limit),
                        },
                    )
                )
                .mappings()
                .all()
            )
        return list(reversed([_row_to_shadow_paper_parity(row) for row in rows]))

    async def shadow_paper_parity_status(
        self,
        signal_name: str,
        signal_type: str,
        *,
        as_of: datetime,
        min_trading_days: int = 20,
        max_target_weight_diff_bps: float = 1.0,
        limit: int = 252,
    ) -> ShadowPaperParityStatus:
        rows = await self.list_shadow_paper_parity(
            signal_name,
            signal_type,
            as_of=as_of,
            limit=limit,
        )
        return _build_shadow_paper_parity_status(
            signal_name,
            signal_type,
            as_of=as_of,
            records=rows,
            min_trading_days=min_trading_days,
            max_target_weight_diff_bps=max_target_weight_diff_bps,
        )
