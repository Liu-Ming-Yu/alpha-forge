"""Domain event serialization regressions."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from quant_platform.core.events import OrderApproved
from quant_platform.infrastructure.event_bus.serialization import deserialize_event, serialize_event


def test_order_approved_round_trips_none_reservation() -> None:
    event = OrderApproved(
        event_id=uuid.uuid4(),
        occurred_at=datetime(2026, 5, 18, 14, 0, tzinfo=UTC),
        order_id=uuid.uuid4(),
        reservation_id=None,
    )

    round_tripped = deserialize_event(serialize_event(event))

    assert isinstance(round_tripped, OrderApproved)
    assert round_tripped.order_id == event.order_id
    assert round_tripped.reservation_id is None
