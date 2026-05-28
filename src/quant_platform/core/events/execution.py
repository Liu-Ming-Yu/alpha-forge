"""Execution and broker-lifecycle domain events."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quant_platform.core.events.base import DomainEvent


@dataclass(frozen=True)
class OrderSubmitted(DomainEvent):
    """An order has been sent to the broker.

    Args:
        order_id: FK to OrderIntent / BrokerOrder.
        broker_order_id: Broker-assigned correlation ID.
    """

    order_id: uuid.UUID
    broker_order_id: str


@dataclass(frozen=True)
class OrderSubmissionUncertain(DomainEvent):
    """An order was transmitted but broker acknowledgement timed out.

    Args:
        order_id: FK to OrderIntent / BrokerOrder.
        broker_order_id: Broker-side order ID assigned before transmission.
        reason: Human-readable uncertainty reason requiring reconciliation.
    """

    order_id: uuid.UUID
    broker_order_id: str
    reason: str


@dataclass(frozen=True)
class OrderFilled(DomainEvent):
    """An order has been completely or partially filled.

    Args:
        order_id: FK to OrderIntent.
        fill_id: FK to the FillEvent.
        filled_quantity: Shares filled in this event.
        fill_price: Execution price.
        is_complete: True if the order is now fully filled.
    """

    order_id: uuid.UUID
    fill_id: uuid.UUID
    filled_quantity: int
    fill_price: Decimal
    is_complete: bool


@dataclass(frozen=True)
class OrderCancelled(DomainEvent):
    """An order has been cancelled (by the system or by the broker).

    Args:
        order_id: FK to OrderIntent.
        reason: Why the order was cancelled.
    """

    order_id: uuid.UUID
    reason: str


@dataclass(frozen=True)
class ReconciliationCompleted(DomainEvent):
    """The execution service has completed a reconciliation cycle.

    Args:
        strategy_run_id: FK to the StrategyRun.
        discrepancies_found: Number of position discrepancies detected.
        discrepancies_resolved: Number resolved automatically.
        requires_operator_action: True if any discrepancy needs manual review.
        cash_drift_usd: Signed difference (ledger − broker) in USD, or None if
            the engine had no ledger reference during this cycle.
    """

    strategy_run_id: uuid.UUID
    discrepancies_found: int
    discrepancies_resolved: int
    requires_operator_action: bool
    is_first_reconciliation: bool = False
    cash_drift_usd: Decimal | None = None


@dataclass(frozen=True)
class CashDriftDetected(DomainEvent):
    """The cash ledger drifted beyond the configured tolerance from broker truth.

    Emitted by ReconciliationEngine when abs(ledger_settled − broker_settled)
    exceeds cash_drift_threshold.  The ledger is reset to broker state immediately
    after this event is recorded.

    Args:
        strategy_run_id: FK to the active StrategyRun.
        ledger_settled_cash: Ledger's view of settled cash at detection time.
        broker_settled_cash: Broker-authoritative settled cash.
        drift_usd: Signed difference ledger − broker (positive = ledger overstates).
    """

    strategy_run_id: uuid.UUID
    ledger_settled_cash: Decimal
    broker_settled_cash: Decimal
    drift_usd: Decimal


@dataclass(frozen=True)
class KillSwitchActivated(DomainEvent):
    """The kill switch has been activated; no new orders will be submitted.

    Args:
        activated_by: Identifier of the operator or automated system that
            triggered the kill switch.
        reason: Human-readable reason.
    """

    activated_by: str
    reason: str


@dataclass(frozen=True)
class SettlementApplied(DomainEvent):
    """A settlement lot has been applied, moving proceeds from unsettled to settled.

    Emitted by AccountStateCoordinator.advance_settlements() each time a lot
    whose settlement_date has arrived is settled.

    Args:
        lot_id: FK to the SettlementLot that was settled.
        fill_id: FK to the originating FillEvent.
        net_proceeds: Dollar amount moved from unsettled_cash to settled_cash.
        strategy_run_id: FK to the StrategyRun for attribution.
    """

    lot_id: uuid.UUID
    fill_id: uuid.UUID
    net_proceeds: Decimal
    strategy_run_id: uuid.UUID


@dataclass(frozen=True)
class BrokerSessionHealthChanged(DomainEvent):
    """The broker gateway connection health has changed.

    Args:
        previous_status: Prior health status string.
        current_status: New health status string.
        detail: Optional detail message from the broker session layer.
    """

    previous_status: str
    current_status: str
    detail: str = ""


@dataclass(frozen=True)
class UnsettledDebitAlert(DomainEvent):
    """The unsettled buy-debit pool has exceeded the configured safety threshold.

    Emitted when the cumulative T+1 buy-side debit exceeds
    ``MAX_UNSETTLED_DEBIT_FRACTION`` of settled cash.  This usually means
    settlement processing has stalled or fills are arriving faster than the
    settlement cycle can clear them.  No trading is halted — this is a warning
    only; operator review is required.

    Args:
        unsettled_buy_debit: Absolute value of the current buy debit pool.
        settled_cash: Current settled cash balance (used as NAV proxy).
        threshold_fraction: The fraction threshold that was exceeded.
    """

    unsettled_buy_debit: Decimal
    settled_cash: Decimal
    threshold_fraction: Decimal


@dataclass(frozen=True)
class UnmatchedFillEvent(DomainEvent):
    """A broker fill arrived with no matching internal order mapping.

    Emitted by the broker gateway when ``commissionReport`` fires for an
    execId whose ib_order_id is not registered in ``_ib_to_instrument``.
    This is an audit record for manual reconciliation — the fill is NOT
    applied to the cash ledger or position store.

    Args:
        ib_order_id: IB integer order identifier from the execution report.
        exec_id: IB execution identifier (matches commissionReport.execId).
        con_id: IB contract identifier from the fill's contract.
    """

    ib_order_id: int
    exec_id: str
    con_id: int


@dataclass(frozen=True)
class OrphanOrderDetected(DomainEvent):
    """An internally tracked order was not found in the broker's open orders.

    Emitted by ``IBGatewayBrokerGateway.fetch_open_orders()`` when an entry
    in ``_submitted`` has been absent from the broker's open-order list for
    longer than ``orphan_ttl_minutes``.  The entry is removed from
    ``_submitted`` after this event is queued.

    Args:
        order_id: Internal UUID of the orphaned order.
        broker_order_id: Last-known IB order ID string.
        acknowledged_at: When the broker originally ack'd the order.
    """

    order_id: uuid.UUID
    broker_order_id: str
    acknowledged_at: datetime


@dataclass(frozen=True)
class ComplianceViolationWarning(DomainEvent):
    """A WARN-severity pre-trade compliance rule was triggered.

    The order was NOT blocked; this event is for operator visibility and audit.

    Args:
        order_id: The order that triggered the warning.
        rule: Compliance rule name (e.g. "PDT_LIMIT", "WASH_SALE").
        detail: Human-readable explanation of the violation.
    """

    order_id: uuid.UUID
    rule: str
    detail: str
