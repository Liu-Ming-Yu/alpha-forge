"""Redis Streams EventBus adapter."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING

import structlog

from quant_platform.infrastructure.event_bus.dead_letters import RedisDeadLetterMixin
from quant_platform.infrastructure.event_bus.redis_consumers import RedisConsumerMixin
from quant_platform.infrastructure.event_bus.redis_dedupe import RedisDedupeMixin
from quant_platform.infrastructure.event_bus.serialization import (
    serialize_event as _serialize_event,
)
from quant_platform.infrastructure.metrics import (
    record_event_publish,
    set_pending_entries_total,
    set_stream_length,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts.redis import AsyncRedisClient
    from quant_platform.core.events import DomainEvent

log = structlog.get_logger(__name__)


def _is_redis_transient(exc: BaseException) -> bool:
    try:
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import TimeoutError as RedisTimeoutError

        return isinstance(exc, (RedisConnectionError, RedisTimeoutError))
    except ImportError:
        return False


class RedisStreamsEventBus(RedisDeadLetterMixin, RedisDedupeMixin, RedisConsumerMixin):
    """Redis Streams EventBus with idempotent publish and consumer-group support."""

    def __init__(
        self,
        redis_url: str,
        *,
        stream_prefix: str = "qp:events",
        maxlen: int = 10000,
        block_ms: int = 1000,
        use_consumer_groups: bool = True,
        group_prefix: str = "qp:cg",
        publish_dedupe_enabled: bool = True,
        dedupe_ttl_seconds: int = 7 * 24 * 60 * 60,
        dead_letter_after_retries: int = 0,
        max_history: int = 10_000,
    ) -> None:
        if not redis_url:
            raise ValueError("redis_url is required for RedisStreamsEventBus")
        self._redis_url = redis_url
        self._stream_prefix = stream_prefix
        self._maxlen = maxlen
        self._block_ms = block_ms
        self._use_consumer_groups = use_consumer_groups
        self._group_prefix = group_prefix
        self._publish_dedupe_enabled = publish_dedupe_enabled
        self._dedupe_ttl_seconds = dedupe_ttl_seconds
        # Retry budget for DLQ (Phase 4.3).  ``0`` disables the DLQ and
        # uses redelivery-only stream handling.
        self._dead_letter_after_retries = dead_letter_after_retries
        self._client: AsyncRedisClient | None = None
        # In-process ring buffer mirrors InMemoryEventBus.history for test introspection.
        self._local_history: deque[DomainEvent] = deque(maxlen=max_history)

    async def _get_client(self) -> AsyncRedisClient:
        if self._client is None:
            from quant_platform.infrastructure.support.redis_factory import (
                create_async_redis_client,
            )

            self._client = create_async_redis_client(
                self._redis_url,
                decode_responses=True,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying Redis client.

        Long-running supervisors should call this once during shutdown so the
        connection pool is released deterministically. Subsequent operations
        will lazily reconnect.
        """
        client = self._client
        self._client = None
        if client is not None:
            import contextlib

            with contextlib.suppress(Exception):
                await client.aclose()

    def _stream_name(self, event_type: type[DomainEvent]) -> str:
        return f"{self._stream_prefix}:{event_type.__name__}"

    def _offset_key(self, event_type: type[DomainEvent], consumer_id: str) -> str:
        return f"{self._stream_name(event_type)}:offset:{consumer_id}"

    def _group_name(self, event_type: type[DomainEvent], consumer_id: str) -> str:
        return f"{self._group_prefix}:{event_type.__name__}:{consumer_id}"

    def _consumer_name(self, consumer_id: str) -> str:
        return consumer_id

    async def _supports_consumer_groups(self, client: AsyncRedisClient) -> bool:
        return all(
            hasattr(client, method_name) for method_name in ("xgroup_create", "xreadgroup", "xack")
        )

    async def _record_pending_for_groups(self, client: AsyncRedisClient, stream: str) -> None:
        """Emit ``quant_event_bus_pending_entries_total`` via ``XPENDING``.

        Best-effort: if ``XINFO`` or ``XPENDING`` fails (e.g. no groups
        created yet) we silently skip; the publish path must never
        fail on observability errors.
        """
        try:
            groups = await client.xinfo_groups(stream)
        except Exception:  # pragma: no cover - advisory only
            return
        for group in groups or []:
            name = group.get("name") if isinstance(group, dict) else None
            if not name:
                continue
            try:
                info = await client.xpending(stream, name)
            except Exception as exc:  # pragma: no cover - advisory only
                log.debug(
                    "redis_streams.pending_scrape_failed",
                    stream=stream,
                    group=str(name),
                    error=str(exc),
                )
                continue
            pending = 0
            if isinstance(info, dict):
                pending = int(info.get("pending", 0) or 0)
            elif isinstance(info, (list, tuple)) and info:
                pending = int(info[0] or 0)
            set_pending_entries_total("redis_streams", stream, str(name), pending)

    async def _ensure_group(
        self,
        *,
        client: AsyncRedisClient,
        stream: str,
        group_name: str,
    ) -> None:
        try:
            await client.xgroup_create(
                stream,
                group_name,
                id="0-0",
                mkstream=True,
            )
        except Exception as exc:
            # Only swallow "group already exists" (BUSYGROUP).  All other
            # Redis errors (auth, OOM, etc.) must propagate so callers know
            # the consumer group was not successfully created.
            detail = str(exc)
            if "BUSYGROUP" not in detail:
                raise
        log.debug(
            "redis_streams.group_ready",
            stream=stream,
            group=group_name,
        )

    @property
    def history(self) -> list[DomainEvent]:
        """In-process published-event log for test introspection."""
        return list(self._local_history)

    async def publish(self, event: DomainEvent) -> None:
        self._local_history.append(event)
        await self._publish_with_retry(event)

    async def _publish_with_retry(
        self,
        event: DomainEvent,
        *,
        _attempts: int = 3,
        _base_delay: float = 0.1,
    ) -> None:
        """Inner publish with exponential back-off on transient Redis errors."""
        client = await self._get_client()
        stream = self._stream_name(type(event))
        if not await self._publish_dedupe_guard(client=client, stream=stream, event=event):
            record_event_publish(backend="redis_streams", outcome="duplicate_skipped")
            log.debug(
                "redis_streams.publish_duplicate_skipped",
                stream=stream,
                event_id=self._event_id(event),
            )
            return
        for attempt in range(_attempts):
            try:
                fields: dict[str, str] = {
                    "payload": _serialize_event(event),
                    "event_id": self._event_id(event),
                }
                # Include correlation_id as a top-level field so consumers can
                # filter related events without deserializing the full payload.
                corr = getattr(event, "correlation_id", None)
                if corr is not None:
                    fields["correlation_id"] = str(corr)
                if self._use_consumer_groups:
                    await client.xadd(stream, fields)
                else:
                    await client.xadd(
                        stream,
                        fields,
                        maxlen=self._maxlen,
                        approximate=True,
                    )
                record_event_publish(backend="redis_streams", outcome="ok")
                return
            except Exception as exc:
                # Do NOT clear the dedupe guard on failure: the xadd may have
                # succeeded even though we didn't receive the ack.  Clearing the
                # guard would allow a duplicate on the next retry.
                record_event_publish(backend="redis_streams", outcome="error")
                is_transient = _is_redis_transient(exc)
                if not is_transient or attempt >= _attempts - 1:
                    raise
                delay = _base_delay * (2**attempt)
                log.warning(
                    "redis_streams.publish_transient_retry",
                    stream=stream,
                    attempt=attempt + 1,
                    retry_delay_s=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
        # Publish-side XLEN observability; authoritative PEL depth comes from
        # the consumer-group scraper via ``event_bus_pending_entries_total``.
        try:
            length = await client.xlen(stream)
            set_stream_length("redis_streams", stream, int(length))
        except Exception as exc:  # pragma: no cover - xlen is advisory only
            log.debug(
                "redis_streams.length_scrape_failed",
                stream=stream,
                error=str(exc),
            )
        if self._use_consumer_groups:
            await self._record_pending_for_groups(client, stream)
