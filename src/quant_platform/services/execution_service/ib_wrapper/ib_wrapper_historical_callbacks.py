"""Historical-data callbacks for the IB wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import asyncio
    from _thread import LockType

    HistoricalBarTuple = tuple[str, float, float, float, float, int]


class IBHistoricalDataCallbackMixin:
    """Historical-data callback methods used by ``_IBWrapper``."""

    _hist_data: dict[int, list[HistoricalBarTuple]]
    _hist_futures: dict[int, asyncio.Future[list[HistoricalBarTuple]]]
    _lifecycle_lock: LockType

    def _resolve(self, future: asyncio.Future[Any], value: object) -> None:
        raise NotImplementedError

    def historicalData(self, reqId: int, bar: object) -> None:
        """Accumulate bars returned by reqHistoricalData under ``reqId``."""
        bar_any = cast("Any", bar)
        with self._lifecycle_lock:
            entries = self._hist_data.setdefault(reqId, [])
            entries.append(
                (
                    str(bar_any.date),
                    float(bar_any.open),
                    float(bar_any.high),
                    float(bar_any.low),
                    float(bar_any.close),
                    int(bar_any.volume) if bar_any.volume is not None else 0,
                )
            )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        """Resolve the future associated with ``reqId`` with the accumulated bars."""
        with self._lifecycle_lock:
            entries = self._hist_data.pop(reqId, [])
            future = self._hist_futures.pop(reqId, None)
        if future is not None:
            self._resolve(future, entries)
