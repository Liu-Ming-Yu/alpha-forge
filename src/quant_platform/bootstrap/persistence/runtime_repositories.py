"""Runtime repository and store composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from quant_platform.infrastructure.event_bus import InMemoryAuditSink, create_event_bus
from quant_platform.infrastructure.performance import build_performance_repository
from quant_platform.infrastructure.postgres.feature_repository import PostgresFeatureRepository
from quant_platform.infrastructure.repositories import (
    InMemoryOrderRepository,
    InMemoryPositionRepository,
)
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository
from quant_platform.services.data_service.stores.parquet_bar_store import ParquetBarStore

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        AuditSink,
        EventBus,
        FeatureRepository,
        HistoricalDataStore,
        OrderRepository,
        PerformanceRepository,
        PositionRepository,
        SignalContributionRepository,
        TextEventProvider,
    )

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RuntimeRepositories:
    event_bus: EventBus
    audit_sink: AuditSink
    order_repo: OrderRepository
    position_repo: PositionRepository
    performance_repo: PerformanceRepository
    feature_repo: FeatureRepository
    signal_contribution_repo: SignalContributionRepository | None
    text_event_store: TextEventProvider
    bar_store: HistoricalDataStore


def build_runtime_repositories(settings: PlatformSettings) -> RuntimeRepositories:
    """Build the storage-facing runtime adapters for one process."""

    from quant_platform.infrastructure.repositories.signal_contributions import (
        build_signal_contribution_repository,
    )
    from quant_platform.services.data_service.text.text_event_store import (
        InMemoryTextEventStore,
        PostgresTextEventStore,
    )

    event_bus = create_event_bus(
        backend=settings.storage.event_bus_backend,
        redis_url=settings.storage.redis_url,
        stream_prefix=settings.storage.redis_stream_prefix,
        stream_maxlen=settings.storage.redis_stream_maxlen,
        stream_block_ms=settings.storage.redis_stream_block_ms,
        stream_use_consumer_groups=settings.storage.redis_stream_use_consumer_groups,
        stream_group_prefix=settings.storage.redis_stream_group_prefix,
        stream_publish_dedupe_enabled=settings.storage.redis_stream_publish_dedupe_enabled,
        stream_dedupe_ttl_seconds=settings.storage.redis_stream_dedupe_ttl_seconds,
        stream_dead_letter_after_retries=settings.storage.redis_stream_dead_letter_after_retries,
    )
    performance_repo = build_performance_repository(settings.storage.postgres_dsn)
    bar_store: HistoricalDataStore = ParquetBarStore(settings.storage.object_store_root)

    if settings.storage.postgres_dsn:
        from quant_platform.infrastructure.postgres.repositories import (
            PostgresAuditSink,
            PostgresOrderRepository,
            PostgresPositionRepository,
            create_pg_engine,
        )

        pg_engine = create_pg_engine(
            settings.storage.postgres_dsn,
            pool_size=settings.storage.postgres_pool_min,
            max_overflow=max(
                0,
                settings.storage.postgres_pool_max - settings.storage.postgres_pool_min,
            ),
        )
        log.info("session.storage_backend", backend="postgres")
        return RuntimeRepositories(
            event_bus=event_bus,
            audit_sink=PostgresAuditSink(pg_engine),
            order_repo=PostgresOrderRepository(pg_engine),
            position_repo=PostgresPositionRepository(pg_engine),
            performance_repo=performance_repo,
            feature_repo=PostgresFeatureRepository(pg_engine),
            signal_contribution_repo=build_signal_contribution_repository(
                settings.storage.postgres_dsn,
                engine=pg_engine,
            ),
            text_event_store=PostgresTextEventStore(pg_engine),
            bar_store=bar_store,
        )

    log.info("session.storage_backend", backend="in_memory")
    return RuntimeRepositories(
        event_bus=event_bus,
        audit_sink=InMemoryAuditSink(),
        order_repo=InMemoryOrderRepository(),
        position_repo=InMemoryPositionRepository(),
        performance_repo=performance_repo,
        feature_repo=InMemoryFeatureRepository(),
        signal_contribution_repo=build_signal_contribution_repository(""),
        text_event_store=InMemoryTextEventStore(),
        bar_store=bar_store,
    )
