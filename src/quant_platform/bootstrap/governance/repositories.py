"""Governance repository composition helpers."""

from __future__ import annotations

from typing import cast

from quant_platform.core.contracts import (
    GovernanceEvidenceRepository,
    ModelRegistryRepository,
)

PerformanceGovernanceRepository = GovernanceEvidenceRepository


def build_performance_repository(dsn: str | None) -> PerformanceGovernanceRepository:
    """Build the configured performance/governance evidence repository."""
    from quant_platform.infrastructure.performance import (
        build_performance_repository as _build_performance_repository,
    )

    return cast("PerformanceGovernanceRepository", _build_performance_repository(dsn))


def build_model_registry(dsn: str) -> ModelRegistryRepository:
    """Build the configured model registry adapter."""
    from quant_platform.infrastructure.postgres.model_registry import (
        build_model_registry as _build_model_registry,
    )

    return cast("ModelRegistryRepository", _build_model_registry(dsn))


__all__ = [
    "PerformanceGovernanceRepository",
    "build_model_registry",
    "build_performance_repository",
]
