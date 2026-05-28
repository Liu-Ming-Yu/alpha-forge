"""Redis Streams consumer-side methods for the EventBus adapter."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

import structlog

from quant_platform.infrastructure.event_bus.serialization import (
    _EVENT_TYPES,
)
from quant_platform.infrastructure.event_bus.serialization import (
    deserialize_event as _deserialize_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from quant_platform.core.contracts.redis import AsyncRedisClient
    from quant_platform.core.events import DomainEvent


class _RedisConsumerState(Protocol):
    _use_consumer_groups: bool
    _block_ms: int
    _dedupe_ttl_seconds: int

    async def _get_client(self) -> AsyncRedisClient: ...
    def _stream_name(self, event_type: type[DomainEvent]) -> str: ...
    def _offset_key(self, event_type: type[DomainEvent], consumer_id: str) -> str: ...
    def _group_name(self, event_type: type[DomainEvent], consumer_id: str) -> str: ...
    def _consumer_name(self, consumer_id: str) -> str: ...
    async def _supports_consumer_groups(self, client: AsyncRedisClient) -> bool: ...
    async def _ensure_group(
        self,
        *,
        client: AsyncRedisClient,
        stream: str,
        group_name: str,
    ) -> None: ...
    def _consume_dedupe_key(
        self,
        *,
        stream: str,
        group_name: str,
        event: DomainEvent,
    ) -> str | None: ...


log = structlog.get_logger(__name__)


class RedisConsumerMixin:
    """Subscribe and recent-event read methods for RedisStreamsEventBus."""

    async def subscribe(
        self: _RedisConsumerState,
        event_type: type[DomainEvent],
        consumer_id: str,
    ) -> AsyncIterator[DomainEvent]:
        client = await self._get_client()
        stream = self._stream_name(event_type)
        if self._use_consumer_groups and await self._supports_consumer_groups(client):
            group_name = self._group_name(event_type, consumer_id)
            consumer_name = self._consumer_name(consumer_id)
            await self._ensure_group(client=client, stream=stream, group_name=group_name)
            next_id: str = "0"
            while True:
                response = await client.xreadgroup(
                    groupname=group_name,
                    consumername=consumer_name,
                    streams={stream: next_id},
                    count=1,
                    block=self._block_ms,
                )
                if not response:
                    if next_id == "0":
                        next_id = ">"
                    await asyncio.sleep(0)
                    continue
                for _, messages in response:
                    if not messages and next_id == "0":
                        next_id = ">"
                        continue
                    for message_id, fields_map in messages:
                        payload = fields_map.get("payload")
                        if not payload:
                            await client.xack(stream, group_name, message_id)
                            continue
                        event = _deserialize_event(payload)
                        dedupe_key = self._consume_dedupe_key(
                            stream=stream,
                            group_name=group_name,
                            event=event,
                        )
                        if dedupe_key and await client.get(dedupe_key):
                            await client.xack(stream, group_name, message_id)
                            continue
                        if not isinstance(event, event_type):
                            await client.xack(stream, group_name, message_id)
                            continue
                        try:
                            yield event
                        except BaseException:
                            log.warning(
                                "redis_streams.subscriber_failed",
                                stream=stream,
                                group=group_name,
                                message_id=message_id,
                            )
                            raise
                        if dedupe_key:
                            await client.set(
                                dedupe_key,
                                "1",
                                ex=self._dedupe_ttl_seconds,
                            )
                        await client.xack(stream, group_name, message_id)
            return

        offset_key = self._offset_key(event_type, consumer_id)
        last_id = await client.get(offset_key)
        if not last_id:
            last_id = "0-0"
        while True:
            response = await client.xread(
                {stream: last_id},
                count=1,
                block=self._block_ms,
            )
            if not response:
                await asyncio.sleep(0)
                continue
            for _, messages in response:
                for message_id, fields_map in messages:
                    payload = fields_map.get("payload")
                    last_id = message_id
                    await client.set(offset_key, message_id)
                    if not payload:
                        continue
                    event = _deserialize_event(payload)
                    if isinstance(event, event_type):
                        yield event

    async def recent_events(
        self: _RedisConsumerState,
        *,
        limit: int = 1000,
        event_type: type[DomainEvent] | None = None,
    ) -> list[DomainEvent]:
        client = await self._get_client()
        if event_type is not None:
            stream = self._stream_name(event_type)
            entries = await client.xrevrange(stream, count=limit)
            events: list[DomainEvent] = []
            for _, fields_map in entries:
                payload = fields_map.get("payload")
                if not payload:
                    continue
                event = _deserialize_event(payload)
                if isinstance(event, event_type):
                    events.append(event)
            events.reverse()
            return events

        per_stream_limit = max(1, min(limit, 500))
        combined: list[DomainEvent] = []
        for cls in _EVENT_TYPES.values():
            stream = self._stream_name(cls)
            entries = await client.xrevrange(stream, count=per_stream_limit)
            for _, fields_map in entries:
                payload = fields_map.get("payload")
                if payload:
                    combined.append(_deserialize_event(payload))
        combined.sort(key=lambda e: e.occurred_at)
        return combined[-limit:]
