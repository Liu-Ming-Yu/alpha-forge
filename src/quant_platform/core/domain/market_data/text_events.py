"""Text event domain models.

A TextEvent represents a structured pointer to a piece of unstructured text
(earnings transcript, SEC filing, news headline, guidance revision) that can
be fed into the LLM text feature extractor.

Invariants:
- TextEvent is immutable (frozen dataclass).
- artifact_uri points to the raw text snapshot in object storage.
- occurred_at is always UTC-aware.
- instrument_id is None for macro-level events (e.g. Fed speeches, SPX news).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime


class TextEventType(StrEnum):
    """Controlled vocabulary for text event sources."""

    EARNINGS_TRANSCRIPT = "earnings_transcript"
    SEC_FILING = "sec_filing"
    NEWS_HEADLINE = "news_headline"
    GUIDANCE_REVISION = "guidance_revision"
    MACRO_COMMENTARY = "macro_commentary"


@dataclass(frozen=True)
class TextEvent:
    """A pointer to a raw text source and its metadata.

    The text content itself is stored externally (S3 or local filesystem) at
    ``artifact_uri``.  The extractor reads from that URI on demand.

    Args:
        event_id: Stable UUID; used for deduplication and cache keying.
        event_type: Controlled vocabulary label.
        occurred_at: UTC timestamp when the event became public.
        source_uri: Raw storage URI of the original text (e.g. SEC EDGAR URL).
        artifact_uri: Storage URI of the archived text snapshot used for
            extraction.  Identical to ``source_uri`` for most sources; may
            differ if the platform downloaded and stored a local copy.
        instrument_id: FK to Instrument if the event is company-specific.
            None for macro events (market commentary, Fed minutes, etc.).
        metadata: Arbitrary key-value pairs for provenance tracking
            (e.g. ``{"ticker": "AAPL", "fiscal_quarter": "Q3-2025"}``).

    Design rule: TextEvent must not carry extracted features or scores.  That
    is the LLMTextFeatureExtractor's responsibility.  TextEvent is a pure
    data pointer.
    """

    event_id: uuid.UUID
    event_type: TextEventType
    occurred_at: datetime
    source_uri: str
    artifact_uri: str
    instrument_id: uuid.UUID | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.occurred_at.tzinfo is None:
            raise ValueError("TextEvent.occurred_at must be timezone-aware")
        if not self.source_uri:
            raise ValueError("TextEvent.source_uri must not be empty")
        if not self.artifact_uri:
            raise ValueError("TextEvent.artifact_uri must not be empty")
        # Replace None metadata with an empty immutable mapping.
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
