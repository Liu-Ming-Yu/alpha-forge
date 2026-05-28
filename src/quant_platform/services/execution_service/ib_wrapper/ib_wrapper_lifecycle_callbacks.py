"""Order lifecycle and fill callbacks for the IB wrapper."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import structlog

from quant_platform.services.execution_service.ib.ib_contract_mapper import contract_con_id
from quant_platform.services.execution_service.ib.ib_lifecycle_mapper import (
    IB_CANCELLED_STATUSES,
    broker_order_from_open_order,
    fill_event_from_pending,
    order_lifecycle_event_from_status,
    parse_execution_time,
    unmatched_fill_event_from_pending,
)

if TYPE_CHECKING:
    import asyncio
    from _thread import LockType

    from ibapi.common import OrderId
    from ibapi.contract import Contract
    from ibapi.order import Order as IBOrder

    from quant_platform.core.domain.orders import BrokerOrder
    from quant_platform.core.domain.orders.lifecycle import BrokerLifecycleEvent
    from quant_platform.services.execution_service.ib.ib_lifecycle_mapper import PendingExecution

log = structlog.get_logger(__name__)


class IBOrderLifecycleCallbackMixin:
    """Order, open-order, execution, and commission callbacks."""

    _cancel_emitted: set[int]
    _cancel_futures: dict[int, asyncio.Future[None]]
    _ib_to_instrument: dict[int, uuid.UUID]
    _ib_to_internal: dict[int, uuid.UUID]
    _lifecycle_lock: LockType
    _lifecycle_queue: list[BrokerLifecycleEvent]
    _open_order_mapping: dict[int, tuple[str, int]]
    _open_orders: list[BrokerOrder]
    _open_orders_done: asyncio.Future[list[BrokerOrder]] | None
    _order_statuses: dict[int, asyncio.Future[str]]
    _pending_execs: dict[str, PendingExecution]

    def _resolve(
        self,
        future: asyncio.Future[str] | asyncio.Future[None] | asyncio.Future[list[BrokerOrder]],
        value: str | None | list[BrokerOrder],
    ) -> None:
        raise NotImplementedError

    def orderStatus(
        self,
        orderId: OrderId,
        status: str,
        filled: Decimal,
        remaining: Decimal,
        avgFillPrice: float,
        permId: int,
        parentId: int,
        lastFillPrice: float,
        clientId: int,
        whyHeld: str,
        mktCapPrice: float,
    ) -> None:
        future = self._order_statuses.get(orderId)
        if future and not future.done():
            self._resolve(future, status)

        if status in IB_CANCELLED_STATUSES:
            cancel_future = self._cancel_futures.get(orderId)
            if cancel_future and not cancel_future.done():
                self._resolve(cancel_future, None)

        with self._lifecycle_lock:
            internal_id = self._ib_to_internal.get(orderId)
            if internal_id is None:
                return
            broker_order_id = str(orderId)
            if status in IB_CANCELLED_STATUSES:
                if orderId in self._cancel_emitted:
                    return
                self._cancel_emitted.add(orderId)
            event = order_lifecycle_event_from_status(
                order_id=internal_id,
                broker_order_id=broker_order_id,
                status=status,
                remaining=remaining,
                occurred_at=datetime.now(tz=UTC),
            )
            if event is not None:
                self._lifecycle_queue.append(event)

    def openOrder(
        self,
        orderId: OrderId,
        contract: Contract,
        order: IBOrder,
        orderState: object,
    ) -> None:
        status = getattr(orderState, "status", None) if orderState else None
        status_str = str(status) if status else "Unknown"
        ref = order.orderRef or ""
        con_id = contract_con_id(contract)
        if not ref:
            log.warning(
                "ib_wrapper.open_order.empty_order_ref",
                ib_order_id=orderId,
                con_id=con_id,
                detail="order has no orderRef; skipping internal mapping",
            )
        self._open_orders.append(
            broker_order_from_open_order(
                order_ref=ref,
                status=status_str,
                broker_order_id=str(orderId),
                observed_at=datetime.now(tz=UTC),
            )
        )
        self._open_order_mapping[orderId] = (ref, con_id)

    def openOrderEnd(self) -> None:
        if self._open_orders_done and not self._open_orders_done.done():
            self._resolve(self._open_orders_done, list(self._open_orders))

    def execDetails(self, reqId: int, contract: Contract, execution: object) -> None:
        execution_any = cast("Any", execution)
        exec_id = str(execution_any.execId)
        ib_order_id = int(execution_any.orderId)
        exec_time = parse_execution_time(
            str(execution_any.time),
            fallback=datetime.now(tz=UTC),
        )
        with self._lifecycle_lock:
            internal_id = self._ib_to_internal.get(ib_order_id)
            self._pending_execs[exec_id] = {
                "internal_id": internal_id,
                "ib_order_id": ib_order_id,
                "exec_id": exec_id,
                "contract": contract,
                "shares": int(execution_any.shares),
                "price": Decimal(str(execution_any.price)),
                "side": str(execution_any.side),
                "time": exec_time,
                "cum_qty": Decimal(str(execution_any.cumQty)),
            }

    def commissionReport(self, commissionReport: object) -> None:
        commission_report_any = cast("Any", commissionReport)
        exec_id = str(commission_report_any.execId)
        with self._lifecycle_lock:
            pending = self._pending_execs.pop(exec_id, None)
            if pending is None:
                return
            internal_id = pending["internal_id"]
            if internal_id is None:
                return

            ib_order_id = pending["ib_order_id"]
            instrument_id = self._ib_to_instrument.get(ib_order_id)
            if instrument_id is None:
                contract = pending.get("contract")
                con_id = contract_con_id(contract) if contract is not None else 0
                log.error(
                    "broker_gateway.commission_report.unmapped_order",
                    ib_order_id=ib_order_id,
                    con_id=con_id,
                    detail=(
                        "fill arrived for order not registered in _ib_to_instrument; "
                        "UnmatchedFillEvent emitted - reconciliation will detect mismatch"
                    ),
                )
                self._lifecycle_queue.append(
                    unmatched_fill_event_from_pending(
                        pending=pending,
                        con_id=con_id,
                        occurred_at=datetime.now(tz=UTC),
                    )
                )
                return

            received_at = datetime.now(tz=UTC)
            self._lifecycle_queue.append(
                fill_event_from_pending(
                    pending=pending,
                    instrument_id=instrument_id,
                    commission=Decimal(str(commission_report_any.commission)),
                    currency=str(commission_report_any.currency or "USD"),
                    received_at=received_at,
                    fill_id=uuid.uuid4(),
                )
            )
