"""In-memory signal-gate operations for the performance repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.domain.production import SignalGateRecord
from quant_platform.infrastructure.performance.inmemory_rows import upsert_sorted
from quant_platform.infrastructure.performance.status import (
    build_signal_gate_status,
    build_text_gate_status,
)

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.domain.production import (
        SignalGateStatus,
        TextSignalGateRecord,
        TextSignalGateStatus,
    )


class InMemorySignalGateMixin:
    """Text and generalized signal-gate methods backed by in-memory lists."""

    _ic: dict[str, list[TextSignalGateRecord]]
    _signals: dict[tuple[str, str], list[SignalGateRecord]]

    async def record_ic(self, record: TextSignalGateRecord) -> None:
        rows = self._ic.setdefault(record.strategy_name, [])
        upsert_sorted(
            rows,
            record,
            identity=lambda row: row.as_of.date(),
            sort_key=lambda row: row.as_of,
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
        rows = list(self._ic.get(strategy_name, []))
        return build_text_gate_status(
            strategy_name,
            as_of=as_of,
            records=rows,
            min_observations=min_observations,
            min_ic=min_ic,
            max_negative_streak=max_negative_streak,
        )

    async def record_signal_observation(self, record: SignalGateRecord) -> None:
        key = (record.signal_type, record.signal_name)
        rows = self._signals.setdefault(key, [])
        upsert_sorted(
            rows,
            record,
            identity=lambda row: row.as_of.date(),
            sort_key=lambda row: row.as_of,
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
        rows = list(self._signals.get((signal_type, signal_name), []))
        return build_signal_gate_status(
            signal_name,
            signal_type,
            as_of=as_of,
            records=rows,
            min_observations=min_observations,
            min_ic=min_ic,
            max_negative_streak=max_negative_streak,
            drawdown_limit=drawdown_limit,
            turnover_limit=turnover_limit,
        )
