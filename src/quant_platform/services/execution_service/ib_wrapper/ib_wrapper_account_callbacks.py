"""Account, position, and time callbacks for the IB wrapper."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Sequence

    from ibapi.contract import Contract


class IBAccountPositionCallbackMixin:
    """Account and position callback methods used by ``_IBWrapper``."""

    _account_active_req_id: int | None
    _account_done: asyncio.Future[dict[str, str]] | None
    _account_values: dict[str, str]
    _positions: list[tuple[str, Contract, Decimal, Decimal]]
    _positions_done: asyncio.Future[list[tuple[str, Contract, Decimal, Decimal]]] | None
    _positions_expected_generation: int
    _positions_generation: int
    _time_future: asyncio.Future[datetime] | None

    def _resolve(
        self,
        future: asyncio.Future[dict[str, str]]
        | asyncio.Future[list[tuple[str, Contract, Decimal, Decimal]]]
        | asyncio.Future[datetime],
        value: dict[str, str] | Sequence[tuple[str, Contract, Decimal, Decimal]] | datetime,
    ) -> None:
        raise NotImplementedError

    def accountSummary(
        self,
        reqId: int,
        account: str,
        tag: str,
        value: str,
        currency: str,
    ) -> None:
        self._account_values[tag] = value

    def accountSummaryEnd(self, reqId: int) -> None:
        if reqId != self._account_active_req_id:
            return  # Stale callback from a prior timed-out request; discard.
        if self._account_done and not self._account_done.done():
            self._resolve(self._account_done, dict(self._account_values))

    def position(self, account: str, contract: Contract, pos: Decimal, avgCost: float) -> None:
        if pos > 0:
            self._positions.append((account, contract, pos, Decimal(str(avgCost))))

    def positionEnd(self) -> None:
        if self._positions_generation != self._positions_expected_generation:
            return  # Stale callback from a prior timed-out request; discard.
        if self._positions_done and not self._positions_done.done():
            self._resolve(self._positions_done, list(self._positions))

    def currentTime(self, time: int) -> None:
        if self._time_future and not self._time_future.done():
            dt = datetime.fromtimestamp(time, tz=UTC)
            self._resolve(self._time_future, dt)
