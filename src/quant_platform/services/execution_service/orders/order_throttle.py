"""Order pacing and throttle for IB Gateway compliance.

Interactive Brokers enforces rate limits at multiple levels:
- Order submission: 50 orders/second sustained, bursts may be rejected.
- API messages: ~50 messages/second across all request types.
- Identical order cancellations: additional penalties apply.

This module implements a token-bucket throttle for order submission that
keeps the system comfortably within IB's limits.

The throttle also acts as the kill switch enforcement point.  When the kill
switch is active, can_submit() returns approved=False for all orders.

Design rules:
- Every code path that calls BrokerGateway.place_order() must first call
  can_submit() on this throttle.
- The kill switch can only be set by a deliberate operator action or by the
  ReconcileBrokerStateController detecting an unresolvable discrepancy.
- The kill switch must remain active until explicitly cleared by an operator.

Token bucket parameters (conservative for a small-portfolio cash account):
- Capacity: 10 tokens (burst allowance).
- Refill rate: 2 tokens/second.
- Cost per submission: 1 token.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.config import ThrottleSettings
from quant_platform.core.contracts import Clock, ExecutionPolicy, TradeDecision
from quant_platform.services.execution_service.orders.order_submission_gate import (
    duplicate_order_reason,
    market_hours_reason,
)
from quant_platform.services.execution_service.orders.order_throttle_kill_switch import (
    OrderThrottleKillSwitchMixin,
)
from quant_platform.services.execution_service.orders.order_token_bucket import OrderTokenBucket

_DEFAULTS = ThrottleSettings()

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.domain.orders import OrderIntent
    from quant_platform.services.execution_service.stores.kill_switch_store import KillSwitchStore
    from quant_platform.services.execution_service.support.trading_calendar import (
        TradingCalendar,
    )


class OrderThrottle(OrderThrottleKillSwitchMixin, ExecutionPolicy):
    """Token-bucket order throttle with kill switch.

    Implements the ExecutionPolicy contract.

    Accepts an optional ``ThrottleSettings`` to configure capacity and
    refill rate; individual kwargs remain available for focused tests.

    Must never:
        Allow order submission when kill_switch_active is True.
        Allow submission when the token bucket is empty.
        Be bypassed for any submission code path.
    """

    def __init__(
        self,
        clock: Clock,
        settings: ThrottleSettings | None = None,
        *,
        capacity: int | None = None,
        refill_rate: float | None = None,
        trading_calendar: TradingCalendar | None = None,
        trading_hours_enforced: bool = False,
        kill_switch_store: KillSwitchStore | None = None,
    ) -> None:
        cfg = settings or _DEFAULTS
        capacity = capacity if capacity is not None else cfg.capacity
        refill_rate = refill_rate if refill_rate is not None else cfg.refill_rate
        self._clock = clock
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._bucket = OrderTokenBucket.full(
            capacity=capacity,
            refill_rate=refill_rate,
            now=clock.now(),
        )
        self._kill_switch_active = False
        self._kill_switch_reason: str = ""
        self._total_submitted: int = 0
        self._calendar = trading_calendar
        self._hours_enforced = trading_hours_enforced
        # Optional durable store.  When present, activate/clear write
        # through and a session-start hydrate restores state after a
        # restart.  Declared ``object`` typed to avoid a hard dependency
        # on ``sqlalchemy`` in environments that only exercise the
        # in-memory path.
        self._kill_switch_store: KillSwitchStore | None = kill_switch_store
        # Tracks order_ids that have already been submitted this session.
        # Prevents double-submission if can_submit() is called twice for the
        # same intent (e.g., from a retry path before record_submission fires).
        self._submitted_ids: set[uuid.UUID] = set()
        if trading_hours_enforced and trading_calendar is None:
            raise ValueError(
                "OrderThrottle: trading_calendar must be supplied when "
                "trading_hours_enforced is True."
            )

    # ------------------------------------------------------------------
    # ExecutionPolicy implementation
    # ------------------------------------------------------------------

    @property
    def kill_switch_active(self) -> bool:
        """True if the kill switch has been activated."""
        return self._kill_switch_active

    def can_submit(self, intent: OrderIntent) -> TradeDecision:
        """Check whether the order may be submitted right now.

        Checks:
        1. Kill switch is not active.
        2. Token bucket has at least one token.

        Args:
            intent: The order seeking submission clearance.

        Returns:
            TradeDecision.approved=True if submission is permitted.
        """
        if self._kill_switch_active:
            return TradeDecision(
                approved=False,
                reason=f"kill switch active: {self._kill_switch_reason}",
                available_cash=Decimal("0"),
                required_cash=Decimal("0"),
            )

        if reason := duplicate_order_reason(intent.order_id, self._submitted_ids):
            return TradeDecision(
                approved=False,
                reason=reason,
                available_cash=Decimal("0"),
                required_cash=Decimal("0"),
            )

        reason = market_hours_reason(
            intent,
            now=self._clock.now(),
            enforced=self._hours_enforced,
            calendar=self._calendar,
        )
        if reason is not None:
            return TradeDecision(
                approved=False,
                reason=reason,
                available_cash=Decimal("0"),
                required_cash=Decimal("0"),
            )

        # Read-only token check: do not mutate _tokens/_last_refill here.
        # Only record_submission() triggers the authoritative refill so that
        # calling can_submit() multiple times in a row (eligibility pre-checks)
        # does not advance the bucket clock and artificially inflate token counts.
        if self._peek_tokens() < 1.0:
            return TradeDecision(
                approved=False,
                reason=f"throttle exhausted: {self._peek_tokens():.2f} tokens available",
                available_cash=Decimal("0"),
                required_cash=Decimal("0"),
            )

        return TradeDecision(
            approved=True,
            reason="throttle ok",
            available_cash=Decimal("0"),
            required_cash=Decimal("0"),
        )

    def record_submission(self, order_id: uuid.UUID) -> None:
        """Consume one token to record an order submission.

        Must be called immediately after BrokerGateway.place_order() succeeds.

        Args:
            order_id: FK to the submitted OrderIntent.
        """
        self._submitted_ids.add(order_id)
        self._bucket.consume(self._clock.now())
        self._total_submitted += 1

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def tokens_available(self) -> float:
        """Current token count (after applying pending refill)."""
        return self._peek_tokens()

    @property
    def total_submitted(self) -> int:
        """Total orders submitted in this session."""
        return self._total_submitted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _peek_tokens(self) -> float:
        """Return theoretical token count WITHOUT mutating state.

        Used by can_submit() so repeated eligibility checks do not advance
        the bucket clock.  Only record_submission() should call _refill().
        """
        return self._bucket.peek(self._clock.now())

    def _refill(self) -> None:
        """Apply pending token accrual and advance the refill timestamp."""
        now = self._clock.now()
        self._bucket.tokens = self._bucket.peek(now)
        self._bucket.last_refill = now
