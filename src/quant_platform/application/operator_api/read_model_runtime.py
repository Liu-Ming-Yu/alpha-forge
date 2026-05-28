"""Runtime account, broker, blotter, and paper-gate read models."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.application.operator_api.read_model_types import (
    BlotterEntry,
    BlotterView,
    BrokerHealthView,
    CashStatusView,
    PaperGateMetricsView,
)
from quant_platform.core.contracts import BrokerHealthStatus
from quant_platform.core.events import (
    DomainEvent,
    KillSwitchActivated,
    OrderApproved,
    OrderFilled,
    OrderRejected,
    ReconciliationCompleted,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.application.operator_api.read_model_types import (
        CashLedgerViewPort,
        ThrottleStateViewPort,
    )
    from quant_platform.core.contracts import (
        BrokerSessionGateway,
        Clock,
        EventBus,
        OrderRepository,
    )


class RuntimeReadModelMixin:
    _clock: Clock
    _cash: CashLedgerViewPort
    _throttle: ThrottleStateViewPort
    _orders: OrderRepository
    _events: EventBus | None
    _account_broker: BrokerSessionGateway | None

    async def _event_history(self, limit: int = 5000) -> list[DomainEvent]:
        if self._events is None:
            return []
        history = getattr(self._events, "history", None)
        if history is not None:
            return list(history)
        reader = getattr(self._events, "recent_events", None)
        if callable(reader):
            events = await reader(limit=limit)
            return list(events)
        return []

    def cash_status(self) -> CashStatusView:
        return CashStatusView(
            as_of=self._clock.now(),
            settled_cash=self._cash.settled_cash,
            unsettled_cash=self._cash.unsettled_cash,
            reserved_cash=self._cash.reserved_cash,
            available_cash=self._cash.available_cash,
        )

    async def broker_health(self) -> BrokerHealthView:
        """Return the live broker health view."""
        connected = False
        status = "unknown"
        detail = ""
        latency_ms: float | None = None
        last_heartbeat_at: datetime | None = None

        if self._account_broker is not None:
            try:
                health = await self._account_broker.health_check()
            except Exception as exc:  # pragma: no cover - connectivity
                status = BrokerHealthStatus.DISCONNECTED.value
                detail = f"health_check raised: {exc}"
            else:
                connected = health.status == BrokerHealthStatus.CONNECTED
                status = health.status.value
                detail = health.detail or ""
                latency_ms = getattr(health, "latency_ms", None)
                last_heartbeat_at = getattr(health, "last_heartbeat_at", None)
        else:
            connected = not self._throttle.kill_switch_active
            status = (
                BrokerHealthStatus.CONNECTED.value
                if connected
                else BrokerHealthStatus.DISCONNECTED.value
            )

        return BrokerHealthView(
            connected=connected,
            kill_switch_active=self._throttle.kill_switch_active,
            kill_switch_reason=self._throttle._kill_switch_reason,
            orders_submitted_this_session=self._throttle.total_submitted,
            throttle_tokens_available=self._throttle.tokens_available,
            status=status,
            detail=detail,
            latency_ms=latency_ms,
            last_heartbeat_at=last_heartbeat_at,
        )

    async def blotter(self, strategy_run_id: uuid.UUID) -> BlotterView:
        intents = await self._orders.list_open_orders(strategy_run_id)
        entries: list[BlotterEntry] = []
        for intent in intents:
            fills = await self._orders.get_fills(intent.order_id)
            total_filled = sum(fill.quantity for fill in fills)

            avg_fill_price: Decimal | None = None
            commission_paid: Decimal | None = None
            if fills:
                total_qty = sum(fill.quantity for fill in fills)
                if total_qty > 0:
                    avg_fill_price = sum(
                        fill.fill_price * fill.quantity for fill in fills
                    ) / Decimal(total_qty)
                commission_paid = sum((fill.commission for fill in fills), Decimal("0"))

            broker_status: str | None = None
            if getattr(intent, "is_terminal", False):
                broker_status = getattr(intent, "terminal_reason", None) or "terminal"
            elif fills:
                broker_status = "filled" if total_filled >= intent.quantity else "partial"
            else:
                broker_status = "pending"

            entries.append(
                BlotterEntry(
                    order_id=intent.order_id,
                    instrument_id=intent.instrument_id,
                    side=intent.side.value,
                    quantity=intent.quantity,
                    order_type=intent.order_type.value,
                    fills_count=len(fills),
                    total_filled=total_filled,
                    avg_fill_price=avg_fill_price,
                    commission_paid=commission_paid,
                    broker_status=broker_status,
                )
            )
        return BlotterView(as_of=self._clock.now(), entries=entries)

    async def paper_gate_metrics(self, strategy_run_id: uuid.UUID) -> PaperGateMetricsView:
        if self._events is None:
            return PaperGateMetricsView(
                as_of=self._clock.now(),
                orders_considered=0,
                reject_rate=Decimal("0"),
                broker_error_rate=Decimal("0"),
                reconcile_discrepancies=0,
                cash_drift_incidents=0,
                stale_reservations=0,
                average_fill_slippage_bps=None,
                fill_quality_summary="event bus unavailable",
            )

        approved = 0
        rejected = 0
        broker_errors = 0
        cash_drift_incidents = 0
        reconcile_discrepancies = 0
        fill_slippages: list[Decimal] = []

        for event in await self._event_history():
            if (
                isinstance(event, ReconciliationCompleted)
                and event.strategy_run_id == strategy_run_id
            ):
                reconcile_discrepancies = event.discrepancies_found
            elif isinstance(event, KillSwitchActivated):
                if "cash drift" in event.reason.lower():
                    cash_drift_incidents += 1
            elif isinstance(event, OrderApproved):
                intent = await self._orders.get_intent(event.order_id)
                if intent and intent.strategy_run_id == strategy_run_id:
                    approved += 1
            elif isinstance(event, OrderRejected):
                intent = await self._orders.get_intent(event.order_id)
                if intent and intent.strategy_run_id == strategy_run_id:
                    rejected += 1
                    if "broker" in event.reason.lower():
                        broker_errors += 1
            elif isinstance(event, OrderFilled):
                intent = await self._orders.get_intent(event.order_id)
                if intent is None or intent.strategy_run_id != strategy_run_id:
                    continue
                if intent.limit_price and intent.limit_price > 0:
                    side_sign = Decimal("1") if intent.side.value == "buy" else Decimal("-1")
                    slippage = (
                        (event.fill_price - intent.limit_price)
                        / intent.limit_price
                        * Decimal("10000")
                        * side_sign
                    )
                    fill_slippages.append(slippage)

        considered = approved + rejected
        reject_rate = Decimal(rejected) / Decimal(considered) if considered else Decimal("0")
        broker_error_rate = Decimal(broker_errors) / Decimal(rejected) if rejected else Decimal("0")

        avg_slippage: Decimal | None = None
        if fill_slippages:
            avg_slippage = sum(fill_slippages, Decimal("0")) / Decimal(len(fill_slippages))

        stale_reservations = len(
            [
                reservation
                for reservation in self._cash.active_reservations()
                if reservation.expires_at <= self._clock.now()
            ]
        )

        summary = (
            "no fills this window"
            if avg_slippage is None
            else f"avg slippage {avg_slippage:.2f} bps vs model price"
        )
        return PaperGateMetricsView(
            as_of=self._clock.now(),
            orders_considered=considered,
            reject_rate=reject_rate,
            broker_error_rate=broker_error_rate,
            reconcile_discrepancies=reconcile_discrepancies,
            cash_drift_incidents=cash_drift_incidents,
            stale_reservations=stale_reservations,
            average_fill_slippage_bps=avg_slippage,
            fill_quality_summary=summary,
        )
