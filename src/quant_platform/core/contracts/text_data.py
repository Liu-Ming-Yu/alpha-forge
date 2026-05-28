"""Text data contracts.

Defines the protocol for storing and retrieving text events used by the
LLM text feature extractor (Phase 5).

Design rules enforced by this contract:
- TextEventProvider is the sole path for accessing text events.
  No service may read the raw text store directly.
- store_event() is idempotent on event_id: storing the same event twice
  must not raise and must not create a duplicate record.
- get_events() returns events sorted by occurred_at ascending.
- Callers must not assume event payloads are in-memory; get_events() may
  perform I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from quant_platform.core.domain.market_data.text_events import TextEvent, TextEventType


@runtime_checkable
class TextEventProvider(Protocol):
    """Storage and retrieval of raw text event pointers.

    Implemented by:
    - ``InMemoryTextEventStore`` (default; dev/CI/backtest)
    - ``PostgresTextEventStore`` (production; wired when
      ``QP__STORAGE__POSTGRES_DSN`` is set)

    Invariants:
    - All timestamps are UTC-aware.
    - ``store_event()`` is idempotent on ``event_id``.
    - ``get_events()`` never returns events outside the requested window.
    - The provider never modifies stored events after initial write.
    """

    async def store_event(self, event: TextEvent) -> None:
        """Persist a text event pointer.

        Args:
            event: The TextEvent to store.

        Idempotency: if a record with the same ``event_id`` already exists the
        call is a no-op.  No exception is raised.

        Raises:
            StorageError: if the underlying store is unavailable.
        """
        ...

    async def get_events(
        self,
        start: datetime,
        end: datetime,
        *,
        instrument_ids: list[UUID] | None = None,
        event_types: list[TextEventType] | None = None,
    ) -> list[TextEvent]:
        """Return text events in the given UTC time window.

        Args:
            start: Inclusive start of the window (UTC-aware).
            end: Exclusive end of the window (UTC-aware).
            instrument_ids: If provided, only return events for these
                instruments.  Macro events (instrument_id=None) are always
                included unless this filter is provided AND it is non-empty.
            event_types: If provided, only return events of these types.

        Returns:
            List of TextEvent sorted by occurred_at ascending.

        Raises:
            ValueError: if start >= end or timestamps are not UTC-aware.
        """
        ...
