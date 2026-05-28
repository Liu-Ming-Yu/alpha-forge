"""In-memory bar store implementing HistoricalDataStore.

Stores MarketBar objects keyed by (instrument_id, bar_seconds, timestamp).
Suitable for testing and small-universe paper trading.  Production should
use the Parquet-backed object storage implementation.

Read-time corporate-action adjustments are shared with ``ParquetBarStore``
via ``corporate_actions.apply_adjustments``; both adapters therefore honour
the ``HistoricalDataStore`` invariant of never returning unadjusted bars.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from quant_platform.services.data_service.reference.corporate_actions import apply_adjustments

if TYPE_CHECKING:
    import uuid
    from datetime import date, datetime

    from quant_platform.core.domain.instruments import CorporateAction
    from quant_platform.core.domain.market_data import MarketBar


class InMemoryBarStore:
    """In-memory HistoricalDataStore for testing and paper trading."""

    def __init__(self) -> None:
        self._bars: dict[tuple[uuid.UUID, int], list[MarketBar]] = defaultdict(list)
        self._bar_ids: set[uuid.UUID] = set()
        self._actions: dict[uuid.UUID, list[CorporateAction]] = defaultdict(list)

    async def get_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        key = (instrument_id, bar_seconds)
        window_bars = [b for b in self._bars.get(key, []) if start <= b.timestamp <= end]
        actions = await self.get_corporate_actions(instrument_id, start.date())
        if actions:
            window_bars = apply_adjustments(window_bars, actions)
        window_bars.sort(key=lambda b: b.timestamp)
        return window_bars

    async def store_bars(self, bars: list[MarketBar]) -> None:
        for bar in bars:
            if bar.bar_id in self._bar_ids:
                continue
            self._bar_ids.add(bar.bar_id)
            key = (bar.instrument_id, bar.bar_seconds)
            self._bars[key].append(bar)

    async def get_corporate_actions(
        self,
        instrument_id: uuid.UUID,
        since: date,
    ) -> list[CorporateAction]:
        """Return all known actions for ``instrument_id``.

        The ``since`` parameter is part of the store protocol; it is
        intentionally ignored so that pre-window CAs still reach the
        adjustment engine. See the matching docstring on
        ``ParquetBarStore.get_corporate_actions``.
        """
        del since
        return list(self._actions.get(instrument_id, []))

    async def store_corporate_action(self, action: CorporateAction) -> None:
        self._actions[action.instrument_id].append(action)
