"""Account state coordinator: broker lifecycle to cash/position truth."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.orders.lifecycle import (
    BrokerFillEvent,
    BrokerLifecycleEvent,
    BrokerOrderCancelled,
    BrokerOrderCompleted,
    BrokerOrderRejected,
)
from quant_platform.services.execution_service.account.account_lifecycle_handlers import (
    AccountLifecycleHandlersMixin,
)
from quant_platform.services.execution_service.account.account_state_storage import (
    AccountStateStorageMixin,
)
from quant_platform.services.execution_service.stores.pending_settlement_store import (
    CompletedOrderHintStore,
    InMemoryCompletedOrderHintStore,
    InMemoryPendingSettlementStore,
    PendingSettlementStore,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.contracts import (
        CashConstraintEngine,
        Clock,
        EventBus,
        OrderRepository,
    )
    from quant_platform.core.domain.settlement import SettlementLot

log = structlog.get_logger(__name__)


@dataclass
class ProcessingResult:
    """Summary of one process_lifecycle_events() call."""

    fills_applied: int = 0
    reservations_released: int = 0
    settlements_projected: int = 0
    errors: int = 0


class AccountStateCoordinator(AccountStateStorageMixin, AccountLifecycleHandlersMixin):
    """Translate broker lifecycle events into CashLedger state and domain events."""

    def __init__(
        self,
        cash_engine: CashConstraintEngine,
        event_bus: EventBus,
        clock: Clock,
        strategy_run_id: uuid.UUID,
        order_repo: OrderRepository | None = None,
        *,
        pending_settlement_store: PendingSettlementStore | None = None,
        completed_order_hint_store: CompletedOrderHintStore | None = None,
        engine_name: str = "default",
        execution_policy: object | None = None,
    ) -> None:
        self._cash = cash_engine
        self._bus = event_bus
        self._clock = clock
        self._strategy_run_id = strategy_run_id
        self._orders = order_repo
        self._execution_policy = execution_policy
        self._pending_lots: list[SettlementLot] = []
        self._completed_order_ids: set[uuid.UUID] = set()
        self._pending_store: PendingSettlementStore = (
            pending_settlement_store or InMemoryPendingSettlementStore()
        )
        self._completed_store: CompletedOrderHintStore = (
            completed_order_hint_store or InMemoryCompletedOrderHintStore()
        )
        self._lot_order_map: dict[uuid.UUID, uuid.UUID] = {}
        self._engine_name = engine_name

    async def process_lifecycle_events(
        self,
        events: list[BrokerLifecycleEvent],
    ) -> ProcessingResult:
        """Process a batch of broker lifecycle events."""
        result = ProcessingResult()
        first_error: Exception | None = None

        for event in events:
            try:
                if isinstance(event, BrokerFillEvent):
                    await self._handle_fill(event, result)
                elif isinstance(event, BrokerOrderCancelled):
                    await self._handle_cancel(event, result)
                elif isinstance(event, BrokerOrderRejected):
                    await self._handle_reject(event, result)
                elif isinstance(event, BrokerOrderCompleted):
                    await self._handle_completed(event, result)
            except Exception as exc:
                log.exception(
                    "coordinator.event_processing_error",
                    event_type=type(event).__name__,
                )
                result.errors += 1
                if first_error is None:
                    first_error = exc

        if result.errors:
            raise RuntimeError(
                "broker lifecycle event processing failed; cycle must halt "
                "and reconcile before applying more state"
            ) from first_error

        return result

    def check_cash_drift(
        self,
        broker_settled: Decimal,
        tolerance: Decimal = Decimal("1.00"),
    ) -> tuple[bool, Decimal]:
        """Compare ledger settled cash against broker reported settled cash."""
        ledger_settled = self._cash.settled_cash
        drift = ledger_settled - broker_settled
        ok = abs(drift) <= tolerance
        if not ok:
            log.warning(
                "coordinator.cash_drift",
                ledger_settled=str(ledger_settled),
                broker_settled=str(broker_settled),
                drift=str(drift),
            )
        return ok, drift


__all__ = ["AccountStateCoordinator", "ProcessingResult"]
