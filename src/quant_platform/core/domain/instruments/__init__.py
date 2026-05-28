"""Compatibility exports for instrument domain models.

Instrument value objects are split into canonical instruments, security-master
metadata, corporate actions, and universe snapshots.  This module remains the
stable import surface for services, adapters, and tests.
"""

from __future__ import annotations

from quant_platform.core.domain.instruments.core import AssetClass, Instrument
from quant_platform.core.domain.instruments.security_master import (
    SecurityMasterQuality,
    SecurityMasterRecord,
    SymbolHistory,
)
from quant_platform.core.domain.instruments.universes import UniverseSnapshot
from quant_platform.core.domain.market_data.corporate_actions import (
    CorporateAction,
    CorporateActionEvent,
    CorporateActionType,
)

__all__ = [
    "AssetClass",
    "CorporateAction",
    "CorporateActionEvent",
    "CorporateActionType",
    "Instrument",
    "SecurityMasterQuality",
    "SecurityMasterRecord",
    "SymbolHistory",
    "UniverseSnapshot",
]
