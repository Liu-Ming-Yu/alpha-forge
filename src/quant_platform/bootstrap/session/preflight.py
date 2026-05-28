"""Runtime session startup preflights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.services.data_service.reference.contract_master import ContractMaster

log = structlog.get_logger(__name__)


def run_sector_mapping_preflight(
    contract_master: ContractMaster,
    sector_map: dict[uuid.UUID, str],
    settings: PlatformSettings,
) -> None:
    """Report and optionally fail on instruments missing a sector label."""
    instruments = contract_master.list_active()
    unmapped = [
        instrument for instrument in instruments if instrument.instrument_id not in sector_map
    ]
    if not unmapped:
        return

    payload = {
        "total_instruments": len(instruments),
        "unmapped_count": len(unmapped),
        "unmapped_symbols": [instrument.symbol for instrument in unmapped],
    }
    if settings.risk.require_sector_mapping:
        log.error("session.sector_mapping.missing", **payload)
        raise ValueError(
            f"risk.require_sector_mapping=True: {len(unmapped)} of "
            f"{len(instruments)} instruments are missing a sector label "
            f"({', '.join(instrument.symbol for instrument in unmapped[:10])}"
            f"{'...' if len(unmapped) > 10 else ''}).  Populate the "
            f"``sector`` field on each Instrument or set "
            f"QP__RISK__REQUIRE_SECTOR_MAPPING=false to opt out."
        )
    log.warning("session.sector_mapping.incomplete", **payload)
