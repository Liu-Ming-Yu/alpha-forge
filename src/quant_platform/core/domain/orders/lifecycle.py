"""Broker lifecycle events.

These represent raw order-lifecycle facts reported by the broker adapter.
They are NOT DomainEvents — they carry broker-native data that has not yet
been validated, attributed to a StrategyRun, or published to the EventBus.

The AccountStateCoordinator translates lifecycle events into:
  - CashLedger mutations (apply_fill, cancel_order, settle_lot)
  - DomainEvent publications (OrderFilled, OrderCancelled, SettlementApplied)

Lifecycle event flow:
    BrokerAdapter.drain_lifecycle_events()
    → AccountStateCoordinator.process_lifecycle_events()
    → CashLedger mutations + EventBus.publish(DomainEvent)

Both IBGatewayBrokerGateway and SimulatedBrokerGateway emit this same set
of lifecycle events, giving the coordinator a single, adapter-agnostic path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.orders import FillEvent


@dataclass(frozen=True)
class BrokerLifecycleEvent:
    """Base class for all broker-reported lifecycle events.

    Must never:
        Carry ibapi-specific types.  All broker types must be translated to
        platform types before being placed in the lifecycle queue.
    """


@dataclass(frozen=True)
class BrokerFillEvent(BrokerLifecycleEvent):
    """A partial or complete fill reported by the broker.

    Args:
        fill: The FillEvent translated from the broker report.
        is_complete: True when the entire order quantity has been executed
            (i.e. remaining == 0).  False for partial fills.

    The is_complete flag determines whether the CashLedger should release
    the cash reservation for this order.  Partial fills deduct settled cash
    but leave the reservation active for the unfilled portion.
    """

    fill: FillEvent
    is_complete: bool


@dataclass(frozen=True)
class BrokerOrderCancelled(BrokerLifecycleEvent):
    """The broker cancelled an open order.

    Emitted when the broker reports status "Cancelled" or equivalent for an
    order that was previously submitted.  The corresponding cash reservation
    must be released immediately.

    Args:
        order_id: Internal UUID of the cancelled order.
        broker_order_id: Broker-assigned order identifier.
        reason: Broker-provided cancellation description.
        occurred_at: UTC timestamp of the broker cancellation report.
    """

    order_id: uuid.UUID
    broker_order_id: str
    reason: str
    occurred_at: datetime


@dataclass(frozen=True)
class BrokerOrderRejected(BrokerLifecycleEvent):
    """The broker rejected an order that was submitted.

    Emitted when the broker reports status "Inactive" or returns error code
    201 (order rejected / insufficient margin) for a submitted order.

    Args:
        order_id: Internal UUID of the rejected order.
        broker_order_id: Broker-assigned order identifier.
        reason: Broker-provided rejection message.
        occurred_at: UTC timestamp of the broker rejection report.
    """

    order_id: uuid.UUID
    broker_order_id: str
    reason: str
    occurred_at: datetime


@dataclass(frozen=True)
class BrokerOrderCompleted(BrokerLifecycleEvent):
    """The broker confirms an order is fully executed (remaining == 0).

    Used on the IB path where fills arrive via execDetails (with
    is_complete=False) and the order-complete signal arrives separately
    via orderStatus(remaining=0, status="Filled").  The coordinator uses
    this event to release the cash reservation for the completed order.

    Not emitted by SimulatedBrokerGateway, which reports completion directly
    on BrokerFillEvent.

    Args:
        order_id: Internal UUID of the completed order.
        broker_order_id: Broker-assigned order identifier.
        occurred_at: UTC timestamp of the broker completion report.
    """

    order_id: uuid.UUID
    broker_order_id: str
    occurred_at: datetime


@dataclass(frozen=True)
class BrokerUnmatchedFill(BrokerLifecycleEvent):
    """A broker fill arrived with no matching internal order mapping.

    Queued by the IB gateway when ``commissionReport`` fires for an execId
    whose ib_order_id is not registered in ``_ib_to_instrument``.  The fill
    is NOT applied to the cash ledger; the AccountStateCoordinator should
    log it and publish an audit ``UnmatchedFillEvent`` to the EventBus.

    Args:
        ib_order_id: IB integer order identifier from the execution report.
        exec_id: IB execution identifier.
        con_id: IB contract identifier from the execution's contract.
        occurred_at: UTC timestamp of detection.
    """

    ib_order_id: int
    exec_id: str
    con_id: int
    occurred_at: datetime


@dataclass(frozen=True)
class BrokerOrphanDetected(BrokerLifecycleEvent):
    """An internally tracked submitted order is absent from broker open orders.

    Queued by ``fetch_open_orders()`` when a ``_submitted`` entry has been
    absent from the broker's open-order list past ``orphan_ttl_minutes``.

    Args:
        order_id: Internal UUID of the orphaned order.
        broker_order_id: Last-known IB order ID string.
        acknowledged_at: When the broker originally ack'd the order.
        occurred_at: UTC timestamp of detection.
    """

    order_id: uuid.UUID
    broker_order_id: str
    acknowledged_at: datetime
    occurred_at: datetime
