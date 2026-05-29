"""Compatibility shim — formulaic panel moved to the inner-layer kernel (ADR-011).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.formulaic.panel``.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.formulaic.panel import (
    DERIVED_COLUMNS as DERIVED_COLUMNS,
)
from quant_platform.services.research_service.features.kernel.formulaic.panel import (
    OPTIONAL_INPUT_COLUMNS as OPTIONAL_INPUT_COLUMNS,
)
from quant_platform.services.research_service.features.kernel.formulaic.panel import (
    REQUIRED_INPUT_COLUMNS as REQUIRED_INPUT_COLUMNS,
)
from quant_platform.services.research_service.features.kernel.formulaic.panel import (
    MarketPanel as MarketPanel,
)
from quant_platform.services.research_service.features.kernel.formulaic.panel import (
    build_market_panel as build_market_panel,
)

__all__ = [
    "DERIVED_COLUMNS",
    "MarketPanel",
    "OPTIONAL_INPUT_COLUMNS",
    "REQUIRED_INPUT_COLUMNS",
    "build_market_panel",
]
