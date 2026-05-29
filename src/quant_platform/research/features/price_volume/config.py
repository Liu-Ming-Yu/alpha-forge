"""Compatibility shim — price-volume config moved to the inner-layer kernel (ADR-011).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.price_volume.config``.
The family registration in this package's ``__init__`` re-exports through here.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.price_volume.config import (
    DEFAULT_CONFIG as DEFAULT_CONFIG,
)
from quant_platform.services.research_service.features.kernel.price_volume.config import (
    FEATURE_SET_VERSION as FEATURE_SET_VERSION,
)
from quant_platform.services.research_service.features.kernel.price_volume.config import (
    LOOKBACK_52W_HIGH as LOOKBACK_52W_HIGH,
)
from quant_platform.services.research_service.features.kernel.price_volume.config import (
    LOOKBACK_AMIHUD as LOOKBACK_AMIHUD,
)
from quant_platform.services.research_service.features.kernel.price_volume.config import (
    LOOKBACK_DOLLAR_VOLUME as LOOKBACK_DOLLAR_VOLUME,
)
from quant_platform.services.research_service.features.kernel.price_volume.config import (
    LOOKBACK_HIGH_LOW_RANGE as LOOKBACK_HIGH_LOW_RANGE,
)
from quant_platform.services.research_service.features.kernel.price_volume.config import (
    LOOKBACK_VOLUME_ZSCORE as LOOKBACK_VOLUME_ZSCORE,
)
from quant_platform.services.research_service.features.kernel.price_volume.config import (
    LOOKBACKS_RETURN as LOOKBACKS_RETURN,
)
from quant_platform.services.research_service.features.kernel.price_volume.config import (
    LOOKBACKS_VOL as LOOKBACKS_VOL,
)
from quant_platform.services.research_service.features.kernel.price_volume.config import (
    PriceVolumeConfig as PriceVolumeConfig,
)

__all__ = [
    "DEFAULT_CONFIG",
    "FEATURE_SET_VERSION",
    "LOOKBACK_52W_HIGH",
    "LOOKBACK_AMIHUD",
    "LOOKBACK_DOLLAR_VOLUME",
    "LOOKBACK_HIGH_LOW_RANGE",
    "LOOKBACK_VOLUME_ZSCORE",
    "LOOKBACKS_RETURN",
    "LOOKBACKS_VOL",
    "PriceVolumeConfig",
]
