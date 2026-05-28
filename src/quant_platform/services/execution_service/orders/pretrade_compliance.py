"""Pre-trade compliance checker for U.S. cash equity accounts.

Stateless rule evaluator called by SubmitOrdersControllerImpl before the
throttle check.  Rules are split by severity:

  BLOCK — order is rejected; ComplianceViolationError is raised by the caller.
  WARN  — order proceeds but the violation is logged and emitted as an event.

Current rules:
  CASH_NO_SHORT (BLOCK): short-selling is not permitted in a cash account.
  HALT (BLOCK): instrument is on the operator-configured halt list.
  PDT_LIMIT (WARN): account is below $25k NAV and day-trade count >= 3.
  WASH_SALE (WARN): same instrument sold within the wash-sale lookback window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

import structlog

from quant_platform.core.domain.orders import OrderIntent, OrderSide

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot

log = structlog.get_logger(__name__)

# Pattern Day Trader threshold
_PDT_NAV_THRESHOLD = Decimal("25000")
_PDT_DAY_TRADE_LIMIT = 3


@dataclass(frozen=True)
class ComplianceViolation:
    """A single pre-trade compliance rule violation."""

    rule: str
    severity: Literal["BLOCK", "WARN"]
    detail: str


@dataclass
class DayTradeCounter:
    """Simple in-process counter for day trades in the current session."""

    _count: int = field(default=0)

    def increment(self) -> None:
        self._count += 1

    @property
    def count(self) -> int:
        return self._count


class PreTradeComplianceChecker:
    """Stateless pre-trade compliance checker for U.S. cash equity accounts.

    Args:
        pdt_enabled: Enable Pattern Day Trader detection (WARN only).
        halted_instruments: Set of instrument UUIDs that are currently halted.
        wash_sale_lookback_days: Days to look back for wash-sale risk.
        day_trades_today: Optional DayTradeCounter; if None PDT check is skipped.
        sell_history: Optional mapping of instrument_id -> last sell datetime,
            used for wash-sale lookback.  If None, wash-sale check is skipped.
    """

    def __init__(
        self,
        *,
        pdt_enabled: bool = True,
        halted_instruments: set[uuid.UUID] | None = None,
        wash_sale_lookback_days: int = 30,
        day_trades_today: DayTradeCounter | None = None,
        sell_history: dict[uuid.UUID, datetime] | None = None,
    ) -> None:
        self._pdt_enabled = pdt_enabled
        self._halted = halted_instruments or set()
        self._wash_sale_days = wash_sale_lookback_days
        self._day_trades = day_trades_today
        self._sell_history: dict[uuid.UUID, datetime] = sell_history or {}

    def check(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        positions: dict[uuid.UUID, PositionSnapshot] | None = None,
    ) -> list[ComplianceViolation]:
        """Evaluate all applicable pre-trade rules for the given intent.

        Args:
            intent: The proposed order.
            account: Current account snapshot (for NAV and position lookup).
            positions: Optional pre-built instrument_id → PositionSnapshot map.
                If None, built from account.positions on each call.

        Returns:
            List of ComplianceViolation records.  BLOCK violations mean the
            order must be rejected; WARN violations are logged but do not stop
            submission.
        """
        pos_map: dict[uuid.UUID, PositionSnapshot] = positions or {
            p.instrument_id: p for p in account.positions
        }
        violations: list[ComplianceViolation] = []
        violations += self._check_halt(intent)
        violations += self._check_no_short_sell(intent, pos_map)
        if intent.side == OrderSide.BUY:
            violations += self._check_pdt(account)
            violations += self._check_wash_sale(intent)
        return violations

    # ------------------------------------------------------------------
    # Individual rule methods
    # ------------------------------------------------------------------

    def _check_halt(self, intent: OrderIntent) -> list[ComplianceViolation]:
        if intent.instrument_id in self._halted:
            return [
                ComplianceViolation(
                    rule="HALT",
                    severity="BLOCK",
                    detail=(
                        f"instrument {intent.instrument_id} is on the halted instruments list; "
                        "order blocked until operator removes it from QP__RISK__HALTED_INSTRUMENTS"
                    ),
                )
            ]
        return []

    def _check_no_short_sell(
        self,
        intent: OrderIntent,
        pos_map: dict[uuid.UUID, PositionSnapshot],
    ) -> list[ComplianceViolation]:
        if intent.side != OrderSide.SELL:
            return []
        pos = pos_map.get(intent.instrument_id)
        available_qty = pos.quantity if pos else 0
        if available_qty < intent.quantity:
            return [
                ComplianceViolation(
                    rule="CASH_NO_SHORT",
                    severity="BLOCK",
                    detail=(
                        f"short selling is not permitted in a cash account: "
                        f"sell quantity={intent.quantity} exceeds held quantity={available_qty} "
                        f"for instrument {intent.instrument_id}"
                    ),
                )
            ]
        return []

    def _check_pdt(self, account: AccountSnapshot) -> list[ComplianceViolation]:
        if not self._pdt_enabled or self._day_trades is None:
            return []
        nav = account.net_asset_value
        if nav < _PDT_NAV_THRESHOLD and self._day_trades.count >= _PDT_DAY_TRADE_LIMIT:
            return [
                ComplianceViolation(
                    rule="PDT_LIMIT",
                    severity="WARN",
                    detail=(
                        f"Pattern Day Trader warning: day_trades_today={self._day_trades.count} "
                        f">= {_PDT_DAY_TRADE_LIMIT} and NAV={nav} < ${_PDT_NAV_THRESHOLD}. "
                        "Order proceeds but operator review is recommended."
                    ),
                )
            ]
        return []

    def _check_wash_sale(self, intent: OrderIntent) -> list[ComplianceViolation]:
        last_sell = self._sell_history.get(intent.instrument_id)
        if last_sell is None:
            return []
        now = datetime.now(tz=UTC)
        lookback = timedelta(days=self._wash_sale_days)
        if (now - last_sell) <= lookback:
            days_ago = (now - last_sell).days
            return [
                ComplianceViolation(
                    rule="WASH_SALE",
                    severity="WARN",
                    detail=(
                        f"wash-sale risk: instrument {intent.instrument_id} was sold "
                        f"{days_ago} day(s) ago (within {self._wash_sale_days}-day lookback). "
                        "Order proceeds; consult your tax adviser."
                    ),
                )
            ]
        return []

    def record_sell(self, instrument_id: uuid.UUID, sold_at: datetime) -> None:
        """Record a completed sell for wash-sale tracking."""
        self._sell_history[instrument_id] = sold_at
