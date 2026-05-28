from __future__ import annotations

from quant_platform.bootstrap.persistence.runtime_repositories import build_runtime_repositories
from quant_platform.config import PlatformSettings
from quant_platform.infrastructure.event_bus import InMemoryAuditSink
from quant_platform.infrastructure.repositories import (
    InMemoryOrderRepository,
    InMemoryPositionRepository,
)
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository


def test_runtime_repositories_default_to_local_adapters() -> None:
    repositories = build_runtime_repositories(PlatformSettings(_env_file=None))

    assert isinstance(repositories.audit_sink, InMemoryAuditSink)
    assert isinstance(repositories.order_repo, InMemoryOrderRepository)
    assert isinstance(repositories.position_repo, InMemoryPositionRepository)
    assert isinstance(repositories.feature_repo, InMemoryFeatureRepository)
