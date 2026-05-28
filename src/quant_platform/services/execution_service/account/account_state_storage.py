"""Settlement-state storage helpers for AccountStateCoordinator."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.events import SettlementApplied
from quant_platform.services.execution_service.stores.pending_settlement_store import (
    hydrate_account_state,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts import CashConstraintEngine, Clock, EventBus
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.settlement import SettlementLot
    from quant_platform.services.execution_service.stores.pending_settlement_store import (
        CompletedOrderHintStore,
        PendingSettlementStore,
    )

log = structlog.get_logger(__name__)


class AccountStateStorageMixin:
    """Durable settlement and completed-order state operations."""

    _completed_order_ids: set[uuid.UUID]
    _bus: EventBus
    _cash: CashConstraintEngine
    _clock: Clock
    _completed_store: CompletedOrderHintStore
    _lot_order_map: dict[uuid.UUID, uuid.UUID]
    _pending_lots: list[SettlementLot]
    _pending_store: PendingSettlementStore
    _strategy_run_id: uuid.UUID

    async def advance_settlements(self) -> int:
        """Settle pending lots whose settlement_date has arrived."""
        settled_count = 0
        remaining: list[SettlementLot] = []
        today = self._clock.today()

        for lot in self._pending_lots:
            if lot.settlement_date <= today:
                try:
                    self._cash.settle_lot(lot)
                    settled_count += 1
                    try:
                        await self._pending_store.delete(lot.lot_id)
                    except Exception:
                        log.exception(
                            "coordinator.pending_lot_delete_error",
                            lot_id=str(lot.lot_id),
                        )
                    self._lot_order_map.pop(lot.lot_id, None)
                    await self._bus.publish(
                        SettlementApplied(
                            event_id=uuid.uuid4(),
                            occurred_at=self._clock.now(),
                            lot_id=lot.lot_id,
                            fill_id=lot.fill_id,
                            net_proceeds=lot.net_proceeds,
                            strategy_run_id=self._strategy_run_id,
                        )
                    )
                    log.info(
                        "coordinator.lot_settled",
                        lot_id=str(lot.lot_id),
                        net_proceeds=str(lot.net_proceeds),
                    )
                except Exception:
                    log.exception("coordinator.settle_lot_error", lot_id=str(lot.lot_id))
                    remaining.append(lot)
            else:
                remaining.append(lot)

        self._pending_lots = remaining
        return settled_count

    @property
    def pending_settlement_lots(self) -> list[SettlementLot]:
        """Lots awaiting settlement (read-only copy)."""
        return list(self._pending_lots)

    def resync_from_broker_snapshot(self, snapshot: AccountSnapshot) -> None:
        """Reset cash-ledger state from a broker-authoritative account snapshot."""
        if hasattr(self._cash, "reset_from_snapshot"):
            self._cash.reset_from_snapshot(snapshot)
        self._pending_lots.clear()
        self._completed_order_ids.clear()
        self._lot_order_map.clear()

    async def purge_durable_state(self) -> None:
        """Drop all durable pending/completed rows for this strategy run."""
        try:
            existing = await self._pending_store.list_all(run_id=self._strategy_run_id)
        except Exception:
            log.exception("coordinator.pending_store.list_failed")
            existing = []
        for lot in existing:
            try:
                await self._pending_store.delete(lot.lot_id)
            except Exception:
                log.exception(
                    "coordinator.pending_store.delete_failed",
                    lot_id=str(lot.lot_id),
                )
        try:
            completed = await self._completed_store.list_all(run_id=self._strategy_run_id)
        except Exception:
            log.exception("coordinator.completed_store.list_failed")
            completed = set()
        for order_id in completed:
            try:
                await self._completed_store.remove(order_id)
            except Exception:
                log.exception(
                    "coordinator.completed_store.remove_failed",
                    order_id=str(order_id),
                )

    async def hydrate(self) -> None:
        """Rehydrate pending-lot and completed-hint state from the stores."""
        state = await hydrate_account_state(
            pending_store=self._pending_store,
            completed_store=self._completed_store,
            run_id=self._strategy_run_id,
        )
        self._pending_lots = list(state.pending_lots)
        self._completed_order_ids = set(state.completed_order_ids)
        self._lot_order_map = {lot.lot_id: uuid.UUID(int=0) for lot in state.pending_lots}
        log.info(
            "coordinator.hydrated",
            pending_lots=len(self._pending_lots),
            completed_order_ids=len(self._completed_order_ids),
        )
