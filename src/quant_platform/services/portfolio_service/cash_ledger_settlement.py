"""Settlement and fill-application behavior for the cash ledger."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.orders import FillEvent, OrderSide
from quant_platform.core.exceptions import (
    PrematureSettlementError,
    SettlementError,
)
from quant_platform.services.portfolio_service.cash_ledger_helpers import (
    compact_uuid_set,
    settlement_lots_for_sell_fills,
)

if TYPE_CHECKING:
    import uuid
    from datetime import date

    from quant_platform.core.contracts import Clock
    from quant_platform.core.domain.settlement import SettlementLot
    from quant_platform.services.portfolio_service.settlement_calendar import SettlementCalendar

# Warn when the T+1 buy-debit pool exceeds this fraction of settled cash.
_MAX_UNSETTLED_DEBIT_FRACTION = Decimal("0.20")

# Compact _settled_lot_ids / _settled_fill_ids when they exceed this size.
_LOT_ID_GC_THRESHOLD = 100_000

_log = structlog.get_logger(__name__)


class CashLedgerSettlementMixin:
    """Fill, settlement projection, and settlement application methods."""

    _buy_t1: bool
    _cal: SettlementCalendar
    _clock: Clock
    _gc_settled_fill_count: int
    _gc_settled_lot_count: int
    _order_to_reservation: dict[uuid.UUID, uuid.UUID]
    _pending_buy_settlements: dict[uuid.UUID, tuple[Decimal, date, uuid.UUID | None]]
    _processed_fill_ids: set[uuid.UUID]
    _settled_cash: Decimal
    _settled_fill_ids: set[uuid.UUID]
    _settled_lot_ids: set[uuid.UUID]
    _submitted_order_ids: set[uuid.UUID]
    _unsettled_buy_debit: Decimal
    _unsettled_sell_proceeds: Decimal

    def release_reservation(self, reservation_id: uuid.UUID, reason: str) -> None:
        raise NotImplementedError

    def project_settlement(
        self,
        fills: list[FillEvent],
    ) -> list[SettlementLot]:
        """Compute projected settlement lots for sell fills."""
        return settlement_lots_for_sell_fills(
            fills,
            settlement_date_for=self._cal.settlement_date,
        )

    def apply_fill(
        self,
        fill: FillEvent,
        is_order_complete: bool = False,
    ) -> None:
        """Update ledger state for a fill event."""
        if fill.fill_id in self._processed_fill_ids:
            return
        self._processed_fill_ids.add(fill.fill_id)

        if fill.side == OrderSide.BUY:
            cost = Decimal(str(fill.quantity)) * fill.fill_price + fill.commission
            if self._buy_t1:
                self._unsettled_buy_debit -= cost
                if (
                    self._settled_cash > Decimal("0")
                    and abs(self._unsettled_buy_debit)
                    > _MAX_UNSETTLED_DEBIT_FRACTION * self._settled_cash
                ):
                    _log.warning(
                        "cash_ledger.unsettled_debit_threshold_exceeded",
                        unsettled_buy_debit=str(self._unsettled_buy_debit),
                        settled_cash=str(self._settled_cash),
                        threshold_fraction=str(_MAX_UNSETTLED_DEBIT_FRACTION),
                    )
                settlement_date = self._cal.settlement_date(fill.executed_at.date())
                res_id = self._order_to_reservation.get(fill.order_id)
                self._pending_buy_settlements[fill.fill_id] = (
                    cost,
                    settlement_date,
                    res_id,
                )
                if is_order_complete:
                    self._submitted_order_ids.discard(fill.order_id)
            else:
                self._settled_cash -= cost
                if is_order_complete:
                    self._submitted_order_ids.discard(fill.order_id)
                    res_id = self._order_to_reservation.get(fill.order_id)
                    if res_id is not None:
                        self.release_reservation(res_id, "order fully filled")
        else:
            net = Decimal(str(fill.quantity)) * fill.fill_price - fill.commission
            self._unsettled_sell_proceeds += net

    def cancel_order(self, order_id: uuid.UUID, reason: str) -> None:
        """Release the cash reservation for a cancelled or rejected order."""
        self._submitted_order_ids.discard(order_id)
        res_id = self._order_to_reservation.get(order_id)
        if res_id is not None:
            self.release_reservation(res_id, reason)

    def settle_lot(self, lot: SettlementLot) -> None:
        """Move settled proceeds from the unsettled pool to settled cash."""
        if lot.lot_id in self._settled_lot_ids:
            return
        if lot.fill_id in self._settled_fill_ids:
            return

        today = self._clock.today()
        if lot.settlement_date > today:
            raise PrematureSettlementError(
                f"lot {lot.lot_id}: settlement_date {lot.settlement_date} "
                f"has not arrived (today={today})"
            )

        if self._unsettled_sell_proceeds < lot.net_proceeds:
            raise SettlementError(
                f"lot {lot.lot_id}: net_proceeds {lot.net_proceeds} exceeds "
                f"unsettled_sell_proceeds {self._unsettled_sell_proceeds}; "
                f"possible duplicate settlement or missing fill"
            )

        self._settled_lot_ids.add(lot.lot_id)
        self._settled_fill_ids.add(lot.fill_id)
        self._unsettled_sell_proceeds -= lot.net_proceeds
        self._settled_cash += lot.net_proceeds

        removed_lots = compact_uuid_set(self._settled_lot_ids, _LOT_ID_GC_THRESHOLD)
        if removed_lots:
            self._gc_settled_lot_count += removed_lots
            _log.info(
                "cash_ledger.gc_settled_lot_ids",
                removed=removed_lots,
                remaining=len(self._settled_lot_ids),
                total_gc=self._gc_settled_lot_count,
            )
        removed_fills = compact_uuid_set(self._settled_fill_ids, _LOT_ID_GC_THRESHOLD)
        if removed_fills:
            self._gc_settled_fill_count += removed_fills

    def settle_pending_buys(self, today: date) -> int:
        """Settle all buy-side pending T+1 fills whose settlement date has arrived."""
        if not self._buy_t1 or not self._pending_buy_settlements:
            return 0
        settled = 0
        for fill_id in list(self._pending_buy_settlements.keys()):
            cost, sd, res_id = self._pending_buy_settlements[fill_id]
            if sd > today:
                continue
            self._unsettled_buy_debit += cost
            self._settled_cash -= cost
            if res_id is not None:
                self.release_reservation(res_id, "buy settled T+1")
            del self._pending_buy_settlements[fill_id]
            settled += 1
        return settled
