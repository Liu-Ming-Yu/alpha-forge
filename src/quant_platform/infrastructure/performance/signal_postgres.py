"""PostgreSQL signal-gate operations for the performance repository."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.core.domain.production import SignalGateRecord
from quant_platform.infrastructure.performance.mappers import row_to_ic as _row_to_ic
from quant_platform.infrastructure.performance.mappers import row_to_signal as _row_to_signal
from quant_platform.infrastructure.performance.status import (
    build_signal_gate_status as _build_signal_gate_status,
)
from quant_platform.infrastructure.performance.status import (
    build_text_gate_status as _build_text_gate_status,
)

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.domain.production import (
        SignalGateStatus,
        TextSignalGateRecord,
        TextSignalGateStatus,
    )


class PostgresSignalGatePerformanceMixin:
    """Text and generalized signal-gate persistence methods."""

    _engine: AsyncEngine

    async def record_ic(self, record: TextSignalGateRecord) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO text_signal_ic_observations
                        (strategy_name, as_of, daily_ic, observations, metadata_json)
                    VALUES (
                        :strategy_name,
                        :as_of,
                        :daily_ic,
                        :observations,
                        CAST(:metadata AS JSONB)
                    )
                    ON CONFLICT (strategy_name, as_of)
                    DO UPDATE SET
                        daily_ic = EXCLUDED.daily_ic,
                        observations = EXCLUDED.observations,
                        metadata_json = EXCLUDED.metadata_json
                """),
                {
                    "strategy_name": record.strategy_name,
                    "as_of": record.as_of,
                    "daily_ic": record.daily_ic,
                    "observations": record.observations,
                    "metadata": json.dumps(record.metadata, default=str),
                },
            )
        await self.record_signal_observation(
            SignalGateRecord(
                signal_name=record.strategy_name,
                signal_type="text",
                as_of=record.as_of,
                daily_ic=record.daily_ic,
                observations=record.observations,
                metadata=record.metadata,
            )
        )

    async def status(
        self,
        strategy_name: str,
        *,
        as_of: datetime,
        min_observations: int = 20,
        min_ic: float = 0.05,
        max_negative_streak: int = 3,
    ) -> TextSignalGateStatus:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT strategy_name, as_of, daily_ic, observations, metadata_json
                            FROM text_signal_ic_observations
                            WHERE strategy_name = :strategy_name AND as_of <= :as_of
                            ORDER BY as_of ASC
                        """),
                        {"strategy_name": strategy_name, "as_of": as_of},
                    )
                )
                .mappings()
                .all()
            )
        records = [_row_to_ic(row) for row in rows]
        return _build_text_gate_status(
            strategy_name,
            as_of=as_of,
            records=records,
            min_observations=min_observations,
            min_ic=min_ic,
            max_negative_streak=max_negative_streak,
        )

    async def record_signal_observation(self, record: SignalGateRecord) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO signal_gate_observations
                        (signal_name, signal_type, as_of, daily_ic, observations,
                         drawdown, turnover, metadata_json)
                    VALUES
                        (:signal_name, :signal_type, :as_of, :daily_ic, :observations,
                         :drawdown, :turnover, CAST(:metadata AS JSONB))
                    ON CONFLICT (signal_name, signal_type, as_of)
                    DO UPDATE SET
                        daily_ic = EXCLUDED.daily_ic,
                        observations = EXCLUDED.observations,
                        drawdown = EXCLUDED.drawdown,
                        turnover = EXCLUDED.turnover,
                        metadata_json = EXCLUDED.metadata_json
                """),
                {
                    "signal_name": record.signal_name,
                    "signal_type": record.signal_type,
                    "as_of": record.as_of,
                    "daily_ic": record.daily_ic,
                    "observations": record.observations,
                    "drawdown": record.drawdown,
                    "turnover": record.turnover,
                    "metadata": json.dumps(record.metadata, default=str),
                },
            )

    async def signal_status(
        self,
        signal_name: str,
        signal_type: str,
        *,
        as_of: datetime,
        min_observations: int = 20,
        min_ic: float = 0.05,
        max_negative_streak: int = 3,
        drawdown_limit: float = -0.10,
        turnover_limit: float = 1.0,
    ) -> SignalGateStatus:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT signal_name, signal_type, as_of, daily_ic, observations,
                                   drawdown, turnover, metadata_json
                            FROM signal_gate_observations
                            WHERE signal_name = :signal_name
                              AND signal_type = :signal_type
                              AND as_of <= :as_of
                            ORDER BY as_of ASC
                        """),
                        {
                            "signal_name": signal_name,
                            "signal_type": signal_type,
                            "as_of": as_of,
                        },
                    )
                )
                .mappings()
                .all()
            )
        records = [_row_to_signal(row) for row in rows]
        return _build_signal_gate_status(
            signal_name,
            signal_type,
            as_of=as_of,
            records=records,
            min_observations=min_observations,
            min_ic=min_ic,
            max_negative_streak=max_negative_streak,
            drawdown_limit=drawdown_limit,
            turnover_limit=turnover_limit,
        )
