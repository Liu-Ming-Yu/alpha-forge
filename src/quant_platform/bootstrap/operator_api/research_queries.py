"""Bootstrap operator query use cases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.operator.queries import OperatorResearchQueryService
from quant_platform.infrastructure.repositories.feature_audit_repository import (
    build_feature_audit_repository,
)

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def build_operator_research_query_service(
    settings: PlatformSettings,
) -> OperatorResearchQueryService:
    repository = (
        build_feature_audit_repository(settings.storage.postgres_dsn)
        if settings.storage.postgres_dsn
        else None
    )
    return OperatorResearchQueryService(
        object_store_root=Path(settings.storage.object_store_root),
        feature_audit_repository=repository,
    )
