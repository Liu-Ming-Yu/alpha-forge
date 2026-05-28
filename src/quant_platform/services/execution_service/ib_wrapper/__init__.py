"""IB callback wrapper for the Interactive Brokers adapter."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import structlog

from quant_platform.services.execution_service.ib_wrapper.ib_wrapper_account_callbacks import (
    IBAccountPositionCallbackMixin,
)
from quant_platform.services.execution_service.ib_wrapper.ib_wrapper_error_callbacks import (
    IBErrorCallbackMixin,
)
from quant_platform.services.execution_service.ib_wrapper.ib_wrapper_historical_callbacks import (
    IBHistoricalDataCallbackMixin,
)
from quant_platform.services.execution_service.ib_wrapper.ib_wrapper_lifecycle_callbacks import (
    IBOrderLifecycleCallbackMixin,
)
from quant_platform.services.execution_service.ib_wrapper.ib_wrapper_news_callbacks import (
    IBNewsCallbackMixin,
)

if TYPE_CHECKING:
    import asyncio
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from ibapi.contract import Contract

    from quant_platform.core.domain.orders import BrokerOrder
    from quant_platform.core.domain.orders.lifecycle import BrokerLifecycleEvent
    from quant_platform.services.execution_service.ib.ib_lifecycle_mapper import PendingExecution

    class EWrapper:
        def __init__(self) -> None: ...

else:
    from ibapi.wrapper import EWrapper

log = structlog.get_logger(__name__)


class _IBWrapper(
    IBErrorCallbackMixin,
    IBOrderLifecycleCallbackMixin,
    IBHistoricalDataCallbackMixin,
    IBNewsCallbackMixin,
    IBAccountPositionCallbackMixin,
    EWrapper,
):
    """Callback handler that bridges ibapi events to asyncio futures."""

    def __init__(self) -> None:
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._connect_event = threading.Event()
        self._connect_error_event = threading.Event()
        self._connect_error: tuple[int, str] | None = None
        self._next_order_id: int | None = None

        self._account_values: dict[str, str] = {}
        self._account_done: asyncio.Future[dict[str, str]] | None = None
        self._account_active_req_id: int | None = None

        self._positions: list[tuple[str, Contract, Decimal, Decimal]] = []
        self._positions_done: (
            asyncio.Future[list[tuple[str, Contract, Decimal, Decimal]]] | None
        ) = None
        self._positions_generation: int = 0
        self._positions_expected_generation: int = 0

        self._order_statuses: dict[int, asyncio.Future[str]] = {}

        self._open_orders: list[BrokerOrder] = []
        self._open_orders_done: asyncio.Future[list[BrokerOrder]] | None = None
        self._open_order_mapping: dict[int, tuple[str, int]] = {}

        self._cancel_futures: dict[int, asyncio.Future[None]] = {}
        self._cancel_emitted: set[int] = set()

        self._time_future: asyncio.Future[datetime] | None = None

        self._hist_data: dict[int, list[tuple[str, float, float, float, float, int]]] = {}
        self._hist_futures: dict[
            int, asyncio.Future[list[tuple[str, float, float, float, float, int]]]
        ] = {}
        self._historical_news: dict[int, list[tuple[str, str, str, str]]] = {}
        self._historical_news_futures: dict[
            int, asyncio.Future[list[tuple[str, str, str, str]]]
        ] = {}
        self._news_article_futures: dict[int, asyncio.Future[tuple[int, str]]] = {}

        self._lifecycle_lock = threading.Lock()
        self._ib_to_internal: dict[int, uuid.UUID] = {}
        self._ib_to_instrument: dict[int, uuid.UUID] = {}
        self._pending_execs: dict[str, PendingExecution] = {}
        self._lifecycle_queue: list[BrokerLifecycleEvent] = []

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _resolve(self, future: asyncio.Future[Any], value: object) -> None:
        if self._loop and not future.done():
            self._loop.call_soon_threadsafe(_set_result_if_pending, future, value)

    def _reject(self, future: asyncio.Future[Any], exc: Exception) -> None:
        if self._loop and not future.done():
            self._loop.call_soon_threadsafe(_set_exception_if_pending, future, exc)

    def connectAck(self) -> None:
        log.info("ib_wrapper.connect_ack")

    def nextValidId(self, orderId: int) -> None:
        self._next_order_id = orderId
        self._connect_event.set()


def _set_result_if_pending(future: asyncio.Future[Any], value: object) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception_if_pending(future: asyncio.Future[Any], exc: Exception) -> None:
    if not future.done():
        future.set_exception(exc)


__all__ = ["_IBWrapper"]
