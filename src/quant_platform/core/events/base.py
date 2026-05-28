"""Base domain event type."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events.

    Args:
        event_id: Stable UUID; used by consumers for deduplication.
        occurred_at: UTC timestamp of the fact this event records.
        correlation_id: Optional UUID linking related events in one workflow.
    """

    event_id: uuid.UUID
    occurred_at: datetime
    # kw_only=True keeps this optional field from blocking required fields in
    # subclasses (Python dataclass inheritance requires non-default fields to
    # precede default fields in the MRO; kw_only fields are exempt).
    correlation_id: uuid.UUID | None = field(default=None, kw_only=True)
