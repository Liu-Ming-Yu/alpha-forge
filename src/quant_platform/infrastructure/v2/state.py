"""In-memory V2 state adapter public facade."""

from __future__ import annotations

from quant_platform.infrastructure.v2.state_datasets import InMemoryDatasetCatalog
from quant_platform.infrastructure.v2.state_execution import InMemoryExecutionQualityRepository
from quant_platform.infrastructure.v2.state_instruments import InMemoryInstrumentRepository
from quant_platform.infrastructure.v2.state_models import InMemoryModelArtifactRepository
from quant_platform.infrastructure.v2.state_orders import InMemoryOrderStateStore
from quant_platform.infrastructure.v2.state_production import (
    InMemoryProductionEvidenceRepository,
)
from quant_platform.infrastructure.v2.state_risk import InMemoryPortfolioRiskModelRepository

__all__ = [
    "InMemoryDatasetCatalog",
    "InMemoryExecutionQualityRepository",
    "InMemoryInstrumentRepository",
    "InMemoryModelArtifactRepository",
    "InMemoryOrderStateStore",
    "InMemoryPortfolioRiskModelRepository",
    "InMemoryProductionEvidenceRepository",
]
