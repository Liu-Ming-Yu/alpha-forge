"""Compatibility shim — feature transforms moved to the inner-layer kernel (ADR-011).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.transforms``. This
re-export keeps the existing ``research.features.transforms`` importers working
without crossing the ``services -> research`` boundary (research, a composition
layer, may import the services inner layer).
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.transforms import (
    BPS_PER_UNIT as BPS_PER_UNIT,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    CALENDAR_DAYS_PER_QUARTER as CALENDAR_DAYS_PER_QUARTER,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    DEFAULT_KEY_COLUMNS as DEFAULT_KEY_COLUMNS,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    KEY_COLUMNS_FUNDAMENTALS as KEY_COLUMNS_FUNDAMENTALS,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    TRADING_DAYS_PER_MONTH as TRADING_DAYS_PER_MONTH,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    TRADING_DAYS_PER_YEAR as TRADING_DAYS_PER_YEAR,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    UNKNOWN_GROUP_SENTINEL as UNKNOWN_GROUP_SENTINEL,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    MinPeriodsPolicy as MinPeriodsPolicy,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    coerce_numeric as coerce_numeric,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    group_by_instrument as group_by_instrument,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    group_pct_change as group_pct_change,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    group_rolling_max as group_rolling_max,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    group_rolling_mean as group_rolling_mean,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    group_rolling_min as group_rolling_min,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    group_rolling_std as group_rolling_std,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    group_rolling_sum as group_rolling_sum,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    group_shift as group_shift,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    ones_like as ones_like,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    safe_div as safe_div,
)

__all__ = [
    "BPS_PER_UNIT",
    "CALENDAR_DAYS_PER_QUARTER",
    "DEFAULT_KEY_COLUMNS",
    "KEY_COLUMNS_FUNDAMENTALS",
    "MinPeriodsPolicy",
    "TRADING_DAYS_PER_MONTH",
    "TRADING_DAYS_PER_YEAR",
    "UNKNOWN_GROUP_SENTINEL",
    "coerce_numeric",
    "group_by_instrument",
    "group_pct_change",
    "group_rolling_max",
    "group_rolling_mean",
    "group_rolling_min",
    "group_rolling_std",
    "group_rolling_sum",
    "group_shift",
    "ones_like",
    "safe_div",
]
