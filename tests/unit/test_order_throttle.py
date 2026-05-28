"""Unit tests for OrderThrottle — token bucket and kill switch."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce
from quant_platform.services.execution_service.orders.order_throttle import OrderThrottle


class _AdvancingClock:
    """Clock that advances on each call to now()."""

    def __init__(self, start: datetime, step: timedelta) -> None:
        self._current = start
        self._step = step

    def now(self) -> datetime:
        t = self._current
        self._current += self._step
        return t

    def today(self) -> date:
        return self._current.date()


_NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=UTC)
_RUN = uuid.uuid4()
_TARGET = uuid.uuid4()
_INSTRUMENT = uuid.uuid4()


def _make_intent() -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET,
        instrument_id=_INSTRUMENT,
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
    )


class TestTokenBucket:
    def test_full_bucket_approves(self) -> None:
        class _FixedClock:
            def now(self) -> datetime:
                return _NOW

            def today(self) -> date:
                return _NOW.date()

        throttle = OrderThrottle(_FixedClock(), capacity=10, refill_rate=2.0)
        decision = throttle.can_submit(_make_intent())
        assert decision.approved

    def test_exhausted_bucket_rejects(self) -> None:
        class _FixedClock:
            def now(self) -> datetime:
                return _NOW

            def today(self) -> date:
                return _NOW.date()

        throttle = OrderThrottle(_FixedClock(), capacity=2, refill_rate=0.0)
        intent = _make_intent()
        throttle.record_submission(intent.order_id)
        throttle.record_submission(intent.order_id)
        decision = throttle.can_submit(_make_intent())
        assert not decision.approved
        assert "throttle" in decision.reason.lower()

    def test_tokens_refill_over_time(self) -> None:
        # Clock advances 1 second per call; refill rate = 2 tokens/sec
        clock = _AdvancingClock(_NOW, timedelta(seconds=1))
        throttle = OrderThrottle(clock, capacity=2, refill_rate=2.0)
        # Exhaust bucket
        throttle.record_submission(uuid.uuid4())
        throttle.record_submission(uuid.uuid4())
        # After 1 second has passed, 2 tokens should have refilled
        decision = throttle.can_submit(_make_intent())
        assert decision.approved


class TestKillSwitch:
    def test_kill_switch_blocks_all_submissions(self) -> None:
        class _FixedClock:
            def now(self) -> datetime:
                return _NOW

            def today(self) -> date:
                return _NOW.date()

        throttle = OrderThrottle(_FixedClock(), capacity=10)
        throttle.activate_kill_switch("test", activated_by="unit_test")
        decision = throttle.can_submit(_make_intent())
        assert not decision.approved
        assert "kill switch" in decision.reason.lower()

    def test_clear_kill_switch_allows_submissions(self) -> None:
        class _FixedClock:
            def now(self) -> datetime:
                return _NOW

            def today(self) -> date:
                return _NOW.date()

        throttle = OrderThrottle(_FixedClock(), capacity=10)
        throttle.activate_kill_switch("test", activated_by="unit_test")
        throttle.clear_kill_switch("operator-1")
        decision = throttle.can_submit(_make_intent())
        assert decision.approved


# ---------------------------------------------------------------------------
# Stream 2 — Double-refill prevention and MOC/LOC market-hours exemption
# ---------------------------------------------------------------------------


class _FixedClock:
    def __init__(self, t: datetime = _NOW) -> None:
        self._t = t

    def now(self) -> datetime:
        return self._t

    def today(self) -> date:
        return self._t.date()


def _make_moc_intent() -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET,
        instrument_id=_INSTRUMENT,
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.MOC,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
    )


def _make_loc_intent() -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET,
        instrument_id=_INSTRUMENT,
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.LOC,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=__import__("decimal", fromlist=["Decimal"]).Decimal("100"),
    )


class TestNoDoubleRefill:
    def test_multiple_can_submit_calls_do_not_consume_tokens(self) -> None:
        """can_submit() must be read-only — repeated calls must not drain tokens."""
        throttle = OrderThrottle(_FixedClock(), capacity=2, refill_rate=0.0)
        intent = _make_intent()
        # Exhaust one token via record_submission.
        throttle.record_submission(uuid.uuid4())
        # 1 token remaining.  Calling can_submit() 3 times must NOT consume it.
        for _ in range(3):
            d = throttle.can_submit(intent)
            assert d.approved, "can_submit must not drain the bucket"
        # Token should still be available for the actual record_submission.
        throttle.record_submission(intent.order_id)
        # Now exhausted.
        assert not throttle.can_submit(_make_intent()).approved

    def test_tokens_visible_after_time_passes_without_submission(self) -> None:
        """tokens_available reflects time-based accrual even without a submission."""
        clock = _AdvancingClock(_NOW, timedelta(seconds=1))
        throttle = OrderThrottle(clock, capacity=10, refill_rate=2.0)
        # Drain completely.
        for _ in range(10):
            throttle.record_submission(uuid.uuid4())
        # tokens_available should reflect ~2 tokens accrued over 1 second
        # (clock advances 1 second on each call).
        available = throttle.tokens_available
        assert available > 0.0


class TestMocLocMarketHoursExemption:
    def test_moc_bypasses_market_closed_gate(self) -> None:
        """MOC orders must be approved even when the market is closed."""
        from unittest.mock import MagicMock

        calendar = MagicMock()
        calendar.is_open.return_value = False  # market is closed

        throttle = OrderThrottle(
            _FixedClock(),
            capacity=10,
            refill_rate=2.0,
            trading_calendar=calendar,
            trading_hours_enforced=True,
        )
        decision = throttle.can_submit(_make_moc_intent())
        assert decision.approved, "MOC should bypass market-closed gate"

    def test_loc_bypasses_market_closed_gate(self) -> None:
        """LOC orders must be approved even when the market is closed."""
        from unittest.mock import MagicMock

        calendar = MagicMock()
        calendar.is_open.return_value = False

        throttle = OrderThrottle(
            _FixedClock(),
            capacity=10,
            refill_rate=2.0,
            trading_calendar=calendar,
            trading_hours_enforced=True,
        )
        decision = throttle.can_submit(_make_loc_intent())
        assert decision.approved, "LOC should bypass market-closed gate"

    def test_market_order_blocked_when_market_closed(self) -> None:
        """A regular MARKET order must be blocked when market is closed."""
        from unittest.mock import MagicMock

        calendar = MagicMock()
        calendar.is_open.return_value = False

        throttle = OrderThrottle(
            _FixedClock(),
            capacity=10,
            refill_rate=2.0,
            trading_calendar=calendar,
            trading_hours_enforced=True,
        )
        decision = throttle.can_submit(_make_intent())
        assert not decision.approved
        assert "market_closed" in decision.reason
