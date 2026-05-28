"""Domain value-object invariant tests.

Covers the type-discipline guarantees added in the audit remediation:

- ``BrokerHealth.latency_ms`` is an ``int`` and rejects negatives.
- ``PortfolioTarget.construction_notes`` is normalised to an immutable
  ``tuple[str, ...]`` via ``__post_init__`` regardless of how the field was
  passed at construction.
- ``OrderStateEvent.idempotency_key`` defaults to ``str(event_id)`` when
  empty / whitespace-only.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.contracts.common import BrokerHealth, BrokerHealthStatus
from quant_platform.core.domain.orders.intent import (
    OrderStateEvent,
    OrderStateEventType,
)
from quant_platform.core.domain.portfolio.targets import PortfolioTarget

_NOW = datetime(2026, 5, 8, tzinfo=UTC)


# ---------------------------------------------------------------------------
# BrokerHealth
# ---------------------------------------------------------------------------


class TestBrokerHealthLatency:
    def test_int_latency_accepted(self) -> None:
        h = BrokerHealth(
            status=BrokerHealthStatus.CONNECTED,
            latency_ms=5,
            last_heartbeat_at=_NOW,
        )
        assert h.latency_ms == 5

    def test_zero_latency_accepted(self) -> None:
        h = BrokerHealth(
            status=BrokerHealthStatus.DISCONNECTED,
            latency_ms=0,
            last_heartbeat_at=_NOW,
        )
        assert h.latency_ms == 0

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValueError, match="latency_ms must be >= 0"):
            BrokerHealth(
                status=BrokerHealthStatus.CONNECTED,
                latency_ms=-1,
                last_heartbeat_at=_NOW,
            )


# ---------------------------------------------------------------------------
# PortfolioTarget.construction_notes
# ---------------------------------------------------------------------------


def _make_target(notes: object) -> PortfolioTarget:
    return PortfolioTarget(
        target_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        as_of=_NOW,
        regime_id=uuid.uuid4(),
        weights={},
        cash_target_weight=Decimal("1.0"),
        construction_notes=notes,  # type: ignore[arg-type]
    )


class TestPortfolioTargetConstructionNotes:
    def test_list_input_is_frozen_to_tuple(self) -> None:
        passed = ["a", "b", "c"]
        target = _make_target(passed)
        assert isinstance(target.construction_notes, tuple)
        assert target.construction_notes == ("a", "b", "c")
        # Mutating the original list must not affect the frozen field.
        passed.append("d")
        assert target.construction_notes == ("a", "b", "c")

    def test_tuple_input_is_preserved(self) -> None:
        target = _make_target(("x", "y"))
        assert target.construction_notes == ("x", "y")

    def test_default_empty(self) -> None:
        target = PortfolioTarget(
            target_id=uuid.uuid4(),
            strategy_run_id=uuid.uuid4(),
            as_of=_NOW,
            regime_id=uuid.uuid4(),
            weights={},
            cash_target_weight=Decimal("1.0"),
        )
        assert target.construction_notes == ()


# ---------------------------------------------------------------------------
# OrderStateEvent.idempotency_key default
# ---------------------------------------------------------------------------


class TestOrderStateEventIdempotency:
    def test_empty_default_falls_back_to_event_id(self) -> None:
        event_id = uuid.uuid4()
        ev = OrderStateEvent(
            event_id=event_id,
            order_id=uuid.uuid4(),
            event_type=OrderStateEventType.ACKNOWLEDGED,
            occurred_at=_NOW,
        )
        assert ev.idempotency_key == str(event_id)

    def test_whitespace_only_falls_back_to_event_id(self) -> None:
        event_id = uuid.uuid4()
        ev = OrderStateEvent(
            event_id=event_id,
            order_id=uuid.uuid4(),
            event_type=OrderStateEventType.ACKNOWLEDGED,
            occurred_at=_NOW,
            idempotency_key="   ",
        )
        assert ev.idempotency_key == str(event_id)

    def test_explicit_key_preserved(self) -> None:
        ev = OrderStateEvent(
            event_id=uuid.uuid4(),
            order_id=uuid.uuid4(),
            event_type=OrderStateEventType.ACKNOWLEDGED,
            occurred_at=_NOW,
            idempotency_key="explicit-key-123",
        )
        assert ev.idempotency_key == "explicit-key-123"
