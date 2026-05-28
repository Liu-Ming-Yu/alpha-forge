"""Bootstrap feature-governance application use cases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.features.governance import FeatureAuditUseCase, SampleBuilder
from quant_platform.infrastructure.repositories.feature_audit_repository import (
    build_feature_audit_repository,
)
from quant_platform.infrastructure.support.artifact_store import FileSystemArtifactStore

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def build_feature_audit_use_case(
    settings: PlatformSettings,
    *,
    sample_builder: SampleBuilder | None = None,
) -> FeatureAuditUseCase:
    """Wire the feature-audit use case to configured adapters."""
    repository = (
        build_feature_audit_repository(settings.storage.postgres_dsn)
        if settings.storage.postgres_dsn
        else None
    )
    return FeatureAuditUseCase(
        object_store_root=Path(settings.storage.object_store_root),
        repository=repository,
        sample_builder=sample_builder,
        artifact_store=FileSystemArtifactStore(settings.storage.object_store_root),
    )
