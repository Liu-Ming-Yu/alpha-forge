"""Publish/consume dedupe helpers for Redis Streams event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from quant_platform.core.contracts.redis import AsyncRedisClient
    from quant_platform.core.events import DomainEvent


class _RedisDedupeState(Protocol):
    _publish_dedupe_enabled: bool
    _dedupe_ttl_seconds: int

    @staticmethod
    def _event_id(event: DomainEvent) -> str: ...


class RedisDedupeMixin:
    """Dedupe keys for publish and consumer-group delivery."""

    @staticmethod
    def _event_id(event: DomainEvent) -> str:
        return str(getattr(event, "event_id", ""))

    async def _publish_dedupe_guard(
        self: _RedisDedupeState,
        *,
        client: AsyncRedisClient,
        stream: str,
        event: DomainEvent,
    ) -> bool:
        if not self._publish_dedupe_enabled:
            return True
        event_id = self._event_id(event)
        if not event_id:
            return True
        inserted = await client.set(
            f"{stream}:published:{event_id}",
            "1",
            nx=True,
            ex=self._dedupe_ttl_seconds,
        )
        return bool(inserted)

    async def _clear_publish_dedupe_guard(
        self: _RedisDedupeState,
        *,
        client: AsyncRedisClient,
        stream: str,
        event: DomainEvent,
    ) -> None:
        if not self._publish_dedupe_enabled:
            return
        event_id = self._event_id(event)
        if event_id:
            await client.delete(f"{stream}:published:{event_id}")

    def _consume_dedupe_key(
        self: _RedisDedupeState,
        *,
        stream: str,
        group_name: str,
        event: DomainEvent,
    ) -> str | None:
        event_id = self._event_id(event)
        if not event_id:
            return None
        return f"{stream}:consumed:{group_name}:{event_id}"
