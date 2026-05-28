"""Broker lifecycle event handlers for account-state coordination."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.orders import OrderSide
from quant_platform.core.events import OrderCancelled, OrderFilled
from quant_platform.telemetry.metrics import observe_fill_latency

if TYPE_CHECKING:
    from quant_platform.core.contracts import CashConstraintEngine, EventBus, OrderRepository
    from quant_platform.core.domain.orders.lifecycle import (
        BrokerFillEvent,
        BrokerOrderCancelled,
        BrokerOrderCompleted,
        BrokerOrderRejected,
    )
    from quant_platform.core.domain.settlement import SettlementLot
    from quant_platform.services.execution_service.account.account_state_coordinator import (
        ProcessingResult,
    )
    from quant_platform.services.execution_service.stores.pending_settlement_store import (
        CompletedOrderHintStore,
        PendingSettlementStore,
    )


log = structlog.get_logger(__name__)


class AccountLifecycleHandlersMixin:
    """Private broker lifecycle handlers for ``AccountStateCoordinator``."""

    _bus: EventBus
    _cash: CashConstraintEngine
    _completed_order_ids: set[uuid.UUID]
    _completed_store: CompletedOrderHintStore
    _engine_name: str
    _execution_policy: object | None
    _lot_order_map: dict[uuid.UUID, uuid.UUID]
    _orders: OrderRepository | None
    _pending_lots: list[SettlementLot]
    _pending_store: PendingSettlementStore
    _strategy_run_id: uuid.UUID

    async def _handle_fill(
        self,
        event: BrokerFillEvent,
        result: ProcessingResult,
    ) -> None:
        fill = event.fill
        is_complete = event.is_complete or fill.order_id in self._completed_order_ids

        if fill.fill_price <= Decimal("0"):
            log.error(
                "coordinator.invalid_fill_price",
                order_id=str(fill.order_id),
                fill_price=str(fill.fill_price),
            )
            if self._execution_policy is not None:
                if hasattr(self._execution_policy, "activate_kill_switch_durable"):
                    import asyncio

                    asyncio.ensure_future(
                        self._execution_policy.activate_kill_switch_durable(
                            "fill_price <= 0", activated_by="account_state_coordinator"
                        )
                    )
                elif hasattr(self._execution_policy, "activate_kill_switch"):
                    self._execution_policy.activate_kill_switch(
                        "fill_price <= 0", activated_by="account_state_coordinator"
                    )
            result.errors += 1
            return

        self._cash.apply_fill(fill, is_order_complete=is_complete)
        result.fills_applied += 1

        if self._orders is not None:
            try:
                intent = await self._orders.get_intent(fill.order_id)
            except Exception:
                intent = None
            if intent is not None:
                submit_ts = getattr(intent, "submitted_at", None) or intent.created_at
                if submit_ts is not None:
                    observe_fill_latency(
                        self._engine_name,
                        max(0.0, (fill.executed_at - submit_ts).total_seconds()),
                    )

        if self._orders is not None:
            await self._orders.save_fill(fill)

        if is_complete:
            self._completed_order_ids.discard(fill.order_id)
            try:
                await self._completed_store.remove(fill.order_id)
            except Exception:
                log.exception(
                    "coordinator.completed_hint_remove_error",
                    order_id=str(fill.order_id),
                )
            result.reservations_released += 1
            await self._mark_terminal(fill.order_id, "fully filled")

        if fill.side == OrderSide.SELL:
            lots = self._cash.project_settlement([fill])
            self._pending_lots.extend(lots)
            result.settlements_projected += len(lots)
            for lot in lots:
                self._lot_order_map[lot.lot_id] = fill.order_id
                try:
                    await self._pending_store.upsert(
                        lot,
                        run_id=self._strategy_run_id,
                        order_id=fill.order_id,
                    )
                except Exception:
                    log.exception(
                        "coordinator.pending_lot_persist_error",
                        lot_id=str(lot.lot_id),
                    )

        await self._bus.publish(
            OrderFilled(
                event_id=uuid.uuid4(),
                occurred_at=fill.executed_at,
                order_id=fill.order_id,
                fill_id=fill.fill_id,
                filled_quantity=fill.quantity,
                fill_price=fill.fill_price,
                is_complete=is_complete,
            )
        )

        log.info(
            "coordinator.fill_applied",
            order_id=str(fill.order_id),
            quantity=fill.quantity,
            price=str(fill.fill_price),
            is_complete=is_complete,
        )

    async def _handle_cancel(
        self,
        event: BrokerOrderCancelled,
        result: ProcessingResult,
    ) -> None:
        self._cash.cancel_order(event.order_id, event.reason)
        result.reservations_released += 1
        await self._mark_terminal(event.order_id, event.reason)

        await self._bus.publish(
            OrderCancelled(
                event_id=uuid.uuid4(),
                occurred_at=event.occurred_at,
                order_id=event.order_id,
                reason=event.reason,
            )
        )

        log.info(
            "coordinator.order_cancelled",
            order_id=str(event.order_id),
            reason=event.reason,
        )

    async def _handle_reject(
        self,
        event: BrokerOrderRejected,
        result: ProcessingResult,
    ) -> None:
        self._cash.cancel_order(event.order_id, f"rejected: {event.reason}")
        result.reservations_released += 1
        await self._mark_terminal(event.order_id, f"rejected: {event.reason}")

        await self._bus.publish(
            OrderCancelled(
                event_id=uuid.uuid4(),
                occurred_at=event.occurred_at,
                order_id=event.order_id,
                reason=f"broker rejected: {event.reason}",
            )
        )

        log.info(
            "coordinator.order_rejected",
            order_id=str(event.order_id),
            reason=event.reason,
        )

    async def _handle_completed(
        self,
        event: BrokerOrderCompleted,
        result: ProcessingResult,
    ) -> None:
        self._completed_order_ids.add(event.order_id)
        try:
            await self._completed_store.add(event.order_id, run_id=self._strategy_run_id)
        except Exception:
            log.exception(
                "coordinator.completed_hint_persist_error",
                order_id=str(event.order_id),
            )
        self._cash.cancel_order(event.order_id, "order fully filled (completed)")
        result.reservations_released += 1
        await self._mark_terminal(event.order_id, "order fully filled (completed)")

        log.info(
            "coordinator.order_completed",
            order_id=str(event.order_id),
        )

    async def _mark_terminal(self, order_id: uuid.UUID, reason: str) -> None:
        if self._orders is None:
            return
        mark_terminal = getattr(self._orders, "mark_terminal", None)
        if mark_terminal is not None:
            await mark_terminal(order_id, reason)
