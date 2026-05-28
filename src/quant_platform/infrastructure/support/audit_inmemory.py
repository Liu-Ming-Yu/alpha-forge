"""In-memory audit sink adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.events import DomainEvent


class InMemoryAuditSink:
    """Append-only in-memory audit log for testing and paper trading."""

    def __init__(self) -> None:
        self._entries: list[tuple[DomainEvent, dict[str, object]]] = []

    async def record(self, event: DomainEvent, context: dict[str, object]) -> None:
        self._entries.append((event, dict(context)))

    @property
    def entries(self) -> list[tuple[DomainEvent, dict[str, object]]]:
        return list(self._entries)

    async def list_events(
        self,
        *,
        event_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        if limit < 1:
            limit = 1
        if limit > 1000:
            limit = 1000
        if offset < 0:
            offset = 0

        def _keep(event: DomainEvent) -> bool:
            if event_type and type(event).__name__ != event_type:
                return False
            occurred_at = getattr(event, "occurred_at", None)
            if since is not None and occurred_at is not None and occurred_at < since:
                return False
            return not (until is not None and occurred_at is not None and occurred_at >= until)

        filtered = [(event, context) for event, context in self._entries if _keep(event)]
        filtered = list(reversed(filtered))
        page = filtered[offset : offset + limit]
        return [
            {
                "audit_id": str(getattr(event, "event_id", "")),
                "event_type": type(event).__name__,
                "event_payload": dict(vars(event)),
                "context": dict(context),
                "recorded_at": str(getattr(event, "occurred_at", "")),
            }
            for event, context in page
        ]
