"""PostgreSQL-backed V2 repository adapters and factory.

This module remains the stable import facade. Concrete repository
implementations live in focused modules so production persistence ownership
does not collapse into one large adapter file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.infrastructure.postgres.repositories import create_pg_engine
from quant_platform.infrastructure.v2.portfolio_json import (
    covariance_to_json as _covariance_to_json,
)
from quant_platform.infrastructure.v2.portfolio_json import (
    factor_exposures_to_json as _factor_exposures_to_json,
)
from quant_platform.infrastructure.v2.portfolio_json import (
    json_to_covariance as _json_to_covariance,
)
from quant_platform.infrastructure.v2.portfolio_json import (
    json_to_factor_exposures as _json_to_factor_exposures,
)
from quant_platform.infrastructure.v2.portfolio_json import (
    json_to_scenarios as _json_to_scenarios,
)
from quant_platform.infrastructure.v2.portfolio_json import (
    scenarios_to_json as _scenarios_to_json,
)
from quant_platform.infrastructure.v2.postgres_datasets import PostgresDatasetCatalog
from quant_platform.infrastructure.v2.postgres_evidence import PostgresProductionEvidenceRepository
from quant_platform.infrastructure.v2.postgres_instruments import PostgresInstrumentRepository
from quant_platform.infrastructure.v2.postgres_mappers import (
    _json,
    _row_to_alpha_report,
    _row_to_feature_dataset,
    _row_to_model_artifact,
    _row_to_operator_action,
    _row_to_operator_api_key,
    _row_to_order_state_event,
    _row_to_quorum,
    _row_to_readiness_snapshot,
    _row_to_risk_model,
    _row_to_security_record,
    _row_to_universe_snapshot,
)
from quant_platform.infrastructure.v2.postgres_models import PostgresModelArtifactRepository
from quant_platform.infrastructure.v2.postgres_orders import (
    PostgresExecutionQualityRepository,
    PostgresOrderStateStore,
)
from quant_platform.infrastructure.v2.postgres_risk import PostgresPortfolioRiskModelRepository
from quant_platform.infrastructure.v2.state import (
    InMemoryDatasetCatalog,
    InMemoryExecutionQualityRepository,
    InMemoryInstrumentRepository,
    InMemoryModelArtifactRepository,
    InMemoryOrderStateStore,
    InMemoryPortfolioRiskModelRepository,
    InMemoryProductionEvidenceRepository,
)

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        OrderStateStore,
        PortfolioRiskModelRepository,
        ProductionEvidenceRepository,
    )


@dataclass(frozen=True)
class V2RepositoryBundle:
    """All V2 repositories needed by production wiring."""

    instrument_repo: object
    dataset_catalog: object
    model_artifacts: object
    risk_models: PortfolioRiskModelRepository
    order_state: OrderStateStore
    execution_quality: object
    production_evidence: ProductionEvidenceRepository
    postgres_backed: bool


def build_v2_repository_bundle(
    settings: PlatformSettings,
    *,
    require_postgres: bool = False,
) -> V2RepositoryBundle:
    """Build V2 repositories, refusing in-memory live wiring unless allowed."""
    dsn = settings.storage.postgres_dsn
    must_be_postgres = require_postgres or (settings.v2.enabled and not settings.allow_dev_defaults)
    if not dsn:
        if must_be_postgres:
            raise RuntimeError(
                "V2 live wiring requires QP__STORAGE__POSTGRES_DSN; "
                "in-memory V2 adapters are only allowed with allow_dev_defaults=true."
            )
        in_memory_evidence: ProductionEvidenceRepository = InMemoryProductionEvidenceRepository()
        return V2RepositoryBundle(
            instrument_repo=InMemoryInstrumentRepository(),
            dataset_catalog=InMemoryDatasetCatalog(),
            model_artifacts=InMemoryModelArtifactRepository(),
            risk_models=InMemoryPortfolioRiskModelRepository(),
            order_state=InMemoryOrderStateStore(),
            execution_quality=InMemoryExecutionQualityRepository(),
            production_evidence=in_memory_evidence,
            postgres_backed=False,
        )

    engine = create_pg_engine(
        dsn,
        pool_size=settings.storage.postgres_pool_min,
        max_overflow=settings.storage.postgres_pool_max - settings.storage.postgres_pool_min,
    )
    postgres_evidence: ProductionEvidenceRepository = PostgresProductionEvidenceRepository(engine)
    return V2RepositoryBundle(
        instrument_repo=PostgresInstrumentRepository(engine),
        dataset_catalog=PostgresDatasetCatalog(engine),
        model_artifacts=PostgresModelArtifactRepository(engine),
        risk_models=PostgresPortfolioRiskModelRepository(engine),
        order_state=PostgresOrderStateStore(engine),
        execution_quality=PostgresExecutionQualityRepository(engine),
        production_evidence=postgres_evidence,
        postgres_backed=True,
    )


def build_dataset_catalog(dsn: str) -> InMemoryDatasetCatalog | PostgresDatasetCatalog:
    """Return a dataset catalog backed by Postgres when a DSN is configured.

    The R-DAT-04 closure path persists vendor-quorum evidence through the
    catalog protocol so the readiness and ``production-candidate`` gates can
    require fresh evidence rather than relying on configuration flags alone.
    """
    if not dsn.strip():
        return InMemoryDatasetCatalog()
    return PostgresDatasetCatalog(create_pg_engine(dsn))


__all__ = [
    "PostgresDatasetCatalog",
    "PostgresExecutionQualityRepository",
    "PostgresInstrumentRepository",
    "PostgresModelArtifactRepository",
    "PostgresOrderStateStore",
    "PostgresPortfolioRiskModelRepository",
    "PostgresProductionEvidenceRepository",
    "V2RepositoryBundle",
    "build_dataset_catalog",
    "build_v2_repository_bundle",
    "_covariance_to_json",
    "_factor_exposures_to_json",
    "_json",
    "_json_to_covariance",
    "_json_to_factor_exposures",
    "_json_to_scenarios",
    "_row_to_alpha_report",
    "_row_to_feature_dataset",
    "_row_to_model_artifact",
    "_row_to_operator_action",
    "_row_to_operator_api_key",
    "_row_to_order_state_event",
    "_row_to_quorum",
    "_row_to_readiness_snapshot",
    "_row_to_risk_model",
    "_row_to_security_record",
    "_row_to_universe_snapshot",
    "_scenarios_to_json",
]
