"""Compatibility exports for market-data domain models.

Market data value objects are split into bar primitives/vendor batches and
versioned dataset/quorum evidence.  This module remains the stable import
surface for services, adapters, and tests.
"""

from __future__ import annotations

from quant_platform.core.domain.market_data.bars import (
    INTRADAY_BAR_SECONDS,
    SUPPORTED_BAR_SECONDS,
    MarketBar,
    VendorBarBatch,
)
from quant_platform.core.domain.market_data.datasets import (
    BarDataset,
    DataLakeLayer,
    DataQualityStatus,
    DatasetQuorumEvidence,
)

__all__ = [
    "SUPPORTED_BAR_SECONDS",
    "INTRADAY_BAR_SECONDS",
    "BarDataset",
    "DataLakeLayer",
    "DataQualityStatus",
    "DatasetQuorumEvidence",
    "MarketBar",
    "VendorBarBatch",
]
