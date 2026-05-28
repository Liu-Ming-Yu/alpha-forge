"""Corporate-action domain models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from quant_platform.core.domain.instruments.security_master import SecurityMasterQuality

if TYPE_CHECKING:
    import uuid
    from datetime import date, datetime


class CorporateActionType(StrEnum):
    SPLIT = "split"
    DIVIDEND = "dividend"
    SPINOFF = "spinoff"
    MERGER = "merger"
    DELISTING = "delisting"


@dataclass(frozen=True)
class CorporateAction:
    """An immutable record of a corporate event affecting position sizing.

    Corporate actions are append-only.  Once recorded they must not be
    modified; corrections are recorded as new entries with a supersedes_id.

    Args:
        action_id: Stable system UUID for this event record.
        instrument_id: The affected instrument.
        action_type: Classification of the corporate event.
        ex_date: The ex-dividend / ex-distribution date.
        record_date: Date used by the registrar to determine entitlement.
        pay_date: Date on which cash or new shares are distributed.
        ratio: For splits/spinoffs, the new-to-old share ratio.
        cash_amount: Per-share cash amount for dividends (zero for non-cash events).
        currency: Currency of cash_amount.
        supersedes_id: If this corrects a prior record, the UUID of that record.
        notes: Free-text annotation from the data provider.
    """

    action_id: uuid.UUID
    instrument_id: uuid.UUID
    action_type: CorporateActionType
    ex_date: date
    record_date: date
    pay_date: date
    ratio: Decimal = Decimal("1")
    cash_amount: Decimal = Decimal("0")
    currency: str = "USD"
    supersedes_id: uuid.UUID | None = None
    notes: str = ""


@dataclass(frozen=True)
class CorporateActionEvent:
    """Vendor-aware, point-in-time corporate-action event.

    ``available_at`` is the timestamp at which downstream research/live code
    could have known the event.  It is the key field that prevents lookahead
    leakage during late-action reprocessing.
    """

    event_id: uuid.UUID
    action: CorporateAction
    source: str
    as_of: datetime
    available_at: datetime
    quality: SecurityMasterQuality = SecurityMasterQuality.APPROVED

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        if self.available_at < self.as_of:
            raise ValueError("available_at must be >= as_of")
