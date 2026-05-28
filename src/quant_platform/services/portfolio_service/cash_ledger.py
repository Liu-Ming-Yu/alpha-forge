"""Settled-cash ledger for the cash-account gating system.

The CashLedger owns in-memory cash state and delegates behavior to focused
mixins: reservation/admission logic and fill/settlement application.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.config import CashSettings
from quant_platform.core.contracts import CashConstraintEngine, Clock
from quant_platform.core.domain.settlement import CashReservation, ReservationStatus
from quant_platform.services.portfolio_service.cash_ledger_reservations import (
    CashLedgerReservationsMixin,
)
from quant_platform.services.portfolio_service.cash_ledger_settlement import (
    _LOT_ID_GC_THRESHOLD,
    _MAX_UNSETTLED_DEBIT_FRACTION,
    CashLedgerSettlementMixin,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterator
    from datetime import date

    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.services.portfolio_service.settlement_calendar import (
        SettlementCalendar,
    )

log = structlog.get_logger(__name__)

__all__ = [
    "CashLedger",
    "_LOT_ID_GC_THRESHOLD",
    "_MAX_UNSETTLED_DEBIT_FRACTION",
]


class CashLedger(
    CashLedgerReservationsMixin,
    CashLedgerSettlementMixin,
    CashConstraintEngine,
):
    """In-process implementation of CashConstraintEngine.

    Seeded from a broker-reconciled AccountSnapshot at session start. State is
    updated incrementally as reservations, fills, and settlement events arrive.
    """

    def __init__(
        self,
        clock: Clock,
        settlement_calendar: SettlementCalendar,
        initial_snapshot: AccountSnapshot,
        settings: CashSettings | None = None,
    ) -> None:
        cfg = settings or CashSettings()
        self._buffer_pct: Decimal = cfg.reservation_buffer_pct
        self._ttl_minutes: int = cfg.reservation_ttl_minutes
        self._buy_t1: bool = cfg.buy_side_t1_settlement

        self._clock = clock
        self._cal = settlement_calendar
        self._settled_cash: Decimal = initial_snapshot.settled_cash
        self._unsettled_sell_proceeds: Decimal = initial_snapshot.unsettled_cash
        self._unsettled_buy_debit: Decimal = Decimal("0")

        self._reservations: dict[uuid.UUID, CashReservation] = {}
        self._order_to_reservation: dict[uuid.UUID, uuid.UUID] = {}
        self._submitted_order_ids: set[uuid.UUID] = set()

        self._processed_fill_ids: set[uuid.UUID] = set()
        self._settled_lot_ids: set[uuid.UUID] = set()
        self._settled_fill_ids: set[uuid.UUID] = set()
        self._gc_settled_lot_count: int = 0
        self._gc_settled_fill_count: int = 0

        self._pending_buy_settlements: dict[uuid.UUID, tuple[Decimal, date, uuid.UUID | None]] = {}

        # Announce the settlement convention at session start so operators can
        # verify against the broker's configured settlement terms (US equities
        # transitioned T+2 → T+1 on 2024-05-28).
        try:
            today = self._clock.now().date()
            convention = settlement_calendar.convention_for(today)
            import structlog

            structlog.get_logger(__name__).info(
                "cash_ledger.settlement_convention",
                convention=convention,
                effective_for=today.isoformat(),
                buy_side_t1_enabled=self._buy_t1,
            )
        except Exception as exc:  # pragma: no cover - protective: logging must not break startup
            log.debug("cash_ledger.settlement_convention_log_failed", error=str(exc))

    @property
    def settled_cash(self) -> Decimal:
        """Total settled cash, including cash earmarked by active reservations."""
        return self._settled_cash

    @property
    def reserved_cash(self) -> Decimal:
        """Sum of all active reservation amounts."""
        return sum(
            (
                r.reserved_amount
                for r in self._reservations.values()
                if r.status == ReservationStatus.ACTIVE
            ),
            Decimal("0"),
        )

    @property
    def available_cash(self) -> Decimal:
        """Settled cash minus all active reservations."""
        return self._settled_cash - self.reserved_cash

    @property
    def unsettled_cash(self) -> Decimal:
        """Sell proceeds that have not reached settlement date."""
        return self._unsettled_sell_proceeds

    def active_reservations(self) -> Iterator[CashReservation]:
        """Yield all currently active cash reservations."""
        for reservation in self._reservations.values():
            if reservation.status == ReservationStatus.ACTIVE:
                yield reservation

    def reset_from_snapshot(self, snapshot: AccountSnapshot) -> None:
        """Hard-reset ledger state from a broker-authoritative account snapshot."""
        self._settled_cash = snapshot.settled_cash
        self._unsettled_sell_proceeds = snapshot.unsettled_cash
        self._unsettled_buy_debit = Decimal("0")
        self._reservations.clear()
        self._order_to_reservation.clear()
        self._submitted_order_ids.clear()
        self._processed_fill_ids.clear()
        self._settled_lot_ids.clear()
        self._settled_fill_ids.clear()
        self._pending_buy_settlements.clear()
        self._gc_settled_lot_count = 0
        self._gc_settled_fill_count = 0
