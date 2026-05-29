"""Compatibility shim — price-volume feature compute moved to the kernel (ADR-011).

Canonical compute now lives in
``quant_platform.services.research_service.features.kernel.price_volume.features``.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.price_volume.features import (
    DEFAULT_TRAINING_FEATURE_NAMES as DEFAULT_TRAINING_FEATURE_NAMES,
)
from quant_platform.services.research_service.features.kernel.price_volume.features import (
    FEATURE_NAMES as FEATURE_NAMES,
)
from quant_platform.services.research_service.features.kernel.price_volume.features import (
    FEATURE_SPECS as FEATURE_SPECS,
)
from quant_platform.services.research_service.features.kernel.price_volume.features import (
    REQUIRED_BAR_COLUMNS as REQUIRED_BAR_COLUMNS,
)
from quant_platform.services.research_service.features.kernel.price_volume.features import (
    REQUIRED_INPUT_COLUMNS as REQUIRED_INPUT_COLUMNS,
)
from quant_platform.services.research_service.features.kernel.price_volume.features import (
    compute_price_volume_features as compute_price_volume_features,
)

__all__ = [
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "REQUIRED_BAR_COLUMNS",
    "REQUIRED_INPUT_COLUMNS",
    "compute_price_volume_features",
]
