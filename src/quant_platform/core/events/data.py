"""Data-service domain events."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from quant_platform.core.domain.market_data.text_events import TextEventType
from quant_platform.core.events.base import DomainEvent


@dataclass(frozen=True)
class MarketBarIngested(DomainEvent):
    """A new market bar has been stored by the data service.

    Args:
        instrument_id: Instrument the bar belongs to.
        bar_id: FK to the new MarketBar record.
        bar_seconds: Resolution of the bar.
    """

    instrument_id: uuid.UUID
    bar_id: uuid.UUID
    bar_seconds: int


@dataclass(frozen=True)
class CorporateActionRecorded(DomainEvent):
    """A corporate action has been stored and prices may need re-adjustment.

    Args:
        instrument_id: Affected instrument.
        action_id: FK to the new CorporateAction record.
    """


@dataclass(frozen=True)
class TextEventIngested(DomainEvent):
    """A text event pointer has been stored by the data service.

    Emitted after ``TextEventProvider.store_event()`` so the research service
    can trigger feature extraction for the new text.

    Args:
        text_event_id: FK to the new TextEvent record.
        event_type: Type of text event (earnings transcript, filing, etc.).
        instrument_id: FK to Instrument if company-specific; None for macro.
    """

    text_event_id: uuid.UUID
    event_type: TextEventType
    instrument_id: uuid.UUID | None
