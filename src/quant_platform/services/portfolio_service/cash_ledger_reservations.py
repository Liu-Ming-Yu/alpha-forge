"""Reservation and cash-admission behavior for the cash ledger."""

from __future__ import annotations

import uuid
from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.contracts import TradeDecision
from quant_platform.core.domain.orders import OrderIntent, OrderSide
from quant_platform.core.domain.settlement import CashReservation, ReservationStatus
from quant_platform.core.exceptions import (
    DataStalenessError,
    DuplicateReservationError,
    InsufficientCashError,
)
from quant_platform.services.portfolio_service.cash_ledger_helpers import (
    price_for_cash_check,
    required_cash_for_buy,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts import Clock
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot


class CashLedgerReservationsMixin:
    """Cash checks, reservations, and reservation lifecycle methods."""

    _buffer_pct: Decimal
    _clock: Clock
    _order_to_reservation: dict[uuid.UUID, uuid.UUID]
    _reservations: dict[uuid.UUID, CashReservation]
    _submitted_order_ids: set[uuid.UUID]
    _ttl_minutes: int

    @property
    def available_cash(self) -> Decimal:
        raise NotImplementedError

    def can_open_order(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
    ) -> TradeDecision:
        """Check whether settled cash or sellable position supports an order."""
        if intent.side == OrderSide.SELL:
            pos = next(
                (p for p in account.positions if p.instrument_id == intent.instrument_id),
                None,
            )
            if pos is None or pos.quantity < intent.quantity:
                available_qty = pos.quantity if pos else 0
                return TradeDecision(
                    approved=False,
                    reason=(
                        f"insufficient position to sell: "
                        f"available={available_qty}, required={intent.quantity}"
                    ),
                    available_cash=self.available_cash,
                    required_cash=Decimal("0"),
                )
            return TradeDecision(
                approved=True,
                reason="sell orders do not require settled cash",
                available_cash=self.available_cash,
                required_cash=Decimal("0"),
            )

        price = price_for_cash_check(intent, account.positions)
        if price is None:
            return TradeDecision(
                approved=False,
                reason="no price available for cash estimation; cannot approve order",
                available_cash=self.available_cash,
                required_cash=Decimal("0"),
            )
        if price <= 0:
            raise DataStalenessError(
                f"market price {price} for instrument {intent.instrument_id} is zero or "
                "negative; refusing to compute notional to prevent unlimited buy approval"
            )

        required = required_cash_for_buy(intent.quantity, price, self._buffer_pct)
        if self.available_cash < required:
            return TradeDecision(
                approved=False,
                reason=(
                    f"insufficient settled cash: "
                    f"available={self.available_cash:.2f}, required={required:.2f}"
                ),
                available_cash=self.available_cash,
                required_cash=required,
            )

        return TradeDecision(
            approved=True,
            reason="sufficient settled cash",
            available_cash=self.available_cash,
            required_cash=required,
        )

    def reserve_cash(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
    ) -> CashReservation:
        """Earmark settled cash for a pending buy order."""
        if intent.side != OrderSide.BUY:
            raise ValueError("reserve_cash is only valid for buy orders")
        if intent.order_id in self._order_to_reservation:
            raise DuplicateReservationError(
                f"ACTIVE reservation already exists for order {intent.order_id}"
            )

        decision = self.can_open_order(intent, account)
        if not decision.approved:
            raise InsufficientCashError(decision.reason)

        now = self._clock.now()
        reservation = CashReservation(
            reservation_id=uuid.uuid4(),
            order_id=intent.order_id,
            reserved_amount=decision.required_cash,
            reserved_at=now,
            expires_at=now + timedelta(minutes=self._ttl_minutes),
            status=ReservationStatus.ACTIVE,
        )
        self._reservations[reservation.reservation_id] = reservation
        self._order_to_reservation[intent.order_id] = reservation.reservation_id
        return reservation

    def release_reservation(
        self,
        reservation_id: uuid.UUID,
        reason: str,
    ) -> None:
        """Release a cash reservation, restoring available cash."""
        reservation = self._reservations.get(reservation_id)
        if reservation is None or reservation.status != ReservationStatus.ACTIVE:
            return

        self._reservations[reservation_id] = CashReservation(
            reservation_id=reservation.reservation_id,
            order_id=reservation.order_id,
            reserved_amount=reservation.reserved_amount,
            reserved_at=reservation.reserved_at,
            expires_at=reservation.expires_at,
            status=ReservationStatus.RELEASED,
            released_at=self._clock.now(),
            release_reason=reason,
        )
        self._order_to_reservation.pop(reservation.order_id, None)

    def is_reservation_active(self, reservation_id: uuid.UUID) -> bool:
        """Return True iff the reservation is currently active."""
        res = self._reservations.get(reservation_id)
        return res is not None and res.status == ReservationStatus.ACTIVE

    def mark_order_submitted(self, order_id: uuid.UUID) -> None:
        """Protect a submitted order's reservation from TTL expiry."""
        self._submitted_order_ids.add(order_id)

    def expire_stale_reservations(self) -> list[uuid.UUID]:
        """Release active pre-submission reservations whose TTL has passed."""
        now = self._clock.now()
        expired: list[uuid.UUID] = []
        for res in list(self._reservations.values()):
            if res.status != ReservationStatus.ACTIVE:
                continue
            if now < res.expires_at:
                continue
            if res.order_id in self._submitted_order_ids:
                continue
            self.release_reservation(res.reservation_id, "reservation TTL expired")
            expired.append(res.reservation_id)
        return expired
