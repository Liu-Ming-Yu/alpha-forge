"""Canonical instrument domain model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid


class AssetClass(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    FUND = "fund"


@dataclass(frozen=True)
class Instrument:
    """Canonical representation of a tradable security.

    Args:
        instrument_id: Stable system UUID; survives ticker renames.
        symbol: Upper-case canonical ticker (e.g. "AAPL").
        exchange: Primary exchange MIC code (e.g. "XNAS").
        asset_class: Broad asset classification.
        currency: Settlement currency ISO code (e.g. "USD").
        lot_size: Minimum tradable quantity (1 for US equities).
        active: False if the instrument has been delisted or suspended.

    Must never contain:
        Broker-specific contract IDs, IBKR conid, margin fields, or
        short-availability data.  Those live in adapter-layer types.
    """

    instrument_id: uuid.UUID
    symbol: str
    exchange: str
    asset_class: AssetClass
    currency: str
    lot_size: int = 1
    active: bool = True
    sector: str | None = None  # GICS sector name (e.g. "Information Technology").
    # None until the data service provides sector mappings; RiskPolicy.evaluate()
    # logs a warning and skips sector-weight checks for instruments without a sector.

    def __post_init__(self) -> None:
        if not self.symbol.isupper():
            raise ValueError(f"symbol must be upper-case, got {self.symbol!r}")
        if self.lot_size < 1:
            raise ValueError("lot_size must be >= 1")
