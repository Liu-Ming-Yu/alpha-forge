"""In-memory contract master for the instrument universe.

The contract master is the single source of truth for which instruments are
tradable and how they map between internal IDs and external symbols.

In a production system this would be backed by PostgreSQL.  This in-memory
version is sufficient for paper trading and tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.domain.instruments import AssetClass, Instrument


class ContractMaster:
    """In-memory instrument registry.

    Args:
        instruments: Initial set of instruments to register.
    """

    def __init__(self, instruments: list[Instrument] | None = None) -> None:
        self._by_id: dict[uuid.UUID, Instrument] = {}
        self._by_symbol: dict[str, Instrument] = {}
        for inst in instruments or []:
            self.register(inst)

    def register(self, instrument: Instrument) -> None:
        """Add or update an instrument in the registry."""
        self._by_id[instrument.instrument_id] = instrument
        self._by_symbol[instrument.symbol] = instrument

    def get_by_id(self, instrument_id: uuid.UUID) -> Instrument | None:
        return self._by_id.get(instrument_id)

    def get_by_symbol(self, symbol: str) -> Instrument | None:
        return self._by_symbol.get(symbol.upper())

    def list_active(self, asset_class: AssetClass | None = None) -> list[Instrument]:
        """Return all active instruments, optionally filtered by asset class."""
        instruments = [i for i in self._by_id.values() if i.active]
        if asset_class is not None:
            instruments = [i for i in instruments if i.asset_class == asset_class]
        return instruments

    def sector_map(self) -> dict[uuid.UUID, str]:
        """Return a mapping of instrument_id → sector for all instruments with sector data."""
        return {i.instrument_id: i.sector for i in self._by_id.values() if i.sector is not None}

    @property
    def size(self) -> int:
        return len(self._by_id)
