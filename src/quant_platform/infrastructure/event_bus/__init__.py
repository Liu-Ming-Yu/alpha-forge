"""Compatibility exports for EventBus and AuditSink adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.infrastructure.event_bus.inmemory import InMemoryEventBus
from quant_platform.infrastructure.event_bus.redis import RedisStreamsEventBus
from quant_platform.infrastructure.event_bus.serialization import (
    deserialize_event as _deserialize_event,
)
from quant_platform.infrastructure.event_bus.serialization import (
    serialize_event as _serialize_event,
)
from quant_platform.infrastructure.metrics import set_pending_entries
from quant_platform.infrastructure.support.audit_inmemory import InMemoryAuditSink

if TYPE_CHECKING:
    from quant_platform.core.contracts import EventBus

__all__ = [
    "InMemoryAuditSink",
    "InMemoryEventBus",
    "RedisStreamsEventBus",
    "_deserialize_event",
    "_serialize_event",
    "create_event_bus",
    "set_pending_entries",
]


def create_event_bus(
    *,
    backend: str = "in_memory",
    redis_url: str = "",
    stream_prefix: str = "qp:events",
    stream_maxlen: int = 10000,
    stream_block_ms: int = 1000,
    stream_use_consumer_groups: bool = True,
    stream_group_prefix: str = "qp:cg",
    stream_publish_dedupe_enabled: bool = True,
    stream_dedupe_ttl_seconds: int = 7 * 24 * 60 * 60,
    stream_dead_letter_after_retries: int = 0,
) -> EventBus:
    """Factory for configured EventBus adapters."""
    if backend == "redis_streams":
        return RedisStreamsEventBus(
            redis_url=redis_url,
            stream_prefix=stream_prefix,
            maxlen=stream_maxlen,
            block_ms=stream_block_ms,
            use_consumer_groups=stream_use_consumer_groups,
            group_prefix=stream_group_prefix,
            publish_dedupe_enabled=stream_publish_dedupe_enabled,
            dedupe_ttl_seconds=stream_dedupe_ttl_seconds,
            dead_letter_after_retries=stream_dead_letter_after_retries,
        )
    return InMemoryEventBus()
