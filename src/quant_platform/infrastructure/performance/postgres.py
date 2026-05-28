"""PostgreSQL performance repository adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.infrastructure.performance.evidence_postgres import (
    PostgresEvidencePerformanceMixin,
)
from quant_platform.infrastructure.performance.observability_postgres import (
    PostgresObservabilityPerformanceMixin,
)
from quant_platform.infrastructure.performance.portfolio_postgres import (
    PostgresPortfolioPerformanceMixin,
)
from quant_platform.infrastructure.performance.shadow_parity_postgres import (
    PostgresShadowPaperParityPerformanceMixin,
)
from quant_platform.infrastructure.performance.signal_postgres import (
    PostgresSignalGatePerformanceMixin,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class PostgresPerformanceRepository(
    PostgresPortfolioPerformanceMixin,
    PostgresObservabilityPerformanceMixin,
    PostgresEvidencePerformanceMixin,
    PostgresSignalGatePerformanceMixin,
    PostgresShadowPaperParityPerformanceMixin,
):
    """PostgreSQL-backed performance and text-promotion repository."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
