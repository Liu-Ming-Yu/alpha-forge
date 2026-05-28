"""Infrastructure contracts: event bus, audit sink, and clock."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence
    from datetime import date, datetime

    from quant_platform.core.events import DomainEvent


@runtime_checkable
class EventBus(Protocol):
    """Publish domain events for cross-service fan-out.

    Delivery is at-least-once.  All consumers must be idempotent.
    Must never:
        Guarantee ordering across different event types.
        Block the caller for more than the configured publish timeout.
    """

    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event to all registered consumers.

        Args:
            event: The event to publish.  Consumers receive a copy.

        Failure semantics:
            Raises EventPublishError on timeout or broker unavailability.
            The caller must decide whether to retry or log-and-continue.
        """
        ...

    def subscribe(
        self,
        event_type: type[DomainEvent],
        consumer_id: str,
    ) -> AsyncIterator[DomainEvent]:
        """Yield events of the given type for the given consumer group.

        Args:
            event_type: The DomainEvent subclass to subscribe to.
            consumer_id: A stable consumer group identifier for offset tracking.
        """
        ...


@runtime_checkable
class AuditSink(Protocol):
    """Append-only audit log for all order and state-change events.

    Must never:
        Allow deletion or modification of existing entries.
        Drop entries even under high load (use a buffered async writer).
    """

    async def record(self, event: DomainEvent, context: dict[str, object]) -> None:
        """Append an audit entry.

        Args:
            event: The domain event to record.
            context: Additional structured context (e.g. operator ID, run ID).
        """
        ...


@runtime_checkable
class Clock(Protocol):
    """Injectable time source for deterministic testing.

    Must never:
        Be bypassed by calling datetime.now() or datetime.utcnow() directly
        in domain or controller code.  Always inject Clock.
    """

    def now(self) -> datetime:
        """Return the current UTC time as a timezone-aware datetime."""
        ...

    def today(self) -> date:
        """Return the current UTC date."""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Typed artifact IO boundary for research/governance evidence."""

    def read_json(self, uri: str) -> Mapping[str, object]:
        """Read one JSON artifact from a stable URI/path."""
        ...

    def write_json(self, uri: str, payload: Mapping[str, object]) -> str:
        """Write one JSON artifact and return its canonical URI/path."""
        ...

    def list_json(self, prefix: str, pattern: str) -> Sequence[Mapping[str, object]]:
        """List JSON artifacts below a prefix using an adapter-defined pattern."""
        ...
