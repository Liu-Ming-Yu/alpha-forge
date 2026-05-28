"""Dead-letter support for the Redis Streams event bus adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import structlog

from quant_platform.infrastructure.metrics import (
    record_dead_letter,
    set_dead_letter_depth,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts.redis import (
        AsyncRedisClient,
        RedisPendingEntry,
    )

log = structlog.get_logger(__name__)


class _RedisDeadLetterState(Protocol):
    _dead_letter_after_retries: int

    async def _get_client(self) -> AsyncRedisClient: ...
    async def _pending_entries_for_group(
        self,
        client: AsyncRedisClient,
        stream: str,
        group_name: str,
    ) -> list[RedisPendingEntry]: ...
    async def _maybe_dead_letter_entry(
        self,
        *,
        client: AsyncRedisClient,
        stream: str,
        dlq_stream: str,
        group_name: str,
        entry: RedisPendingEntry,
        budget: int,
    ) -> int: ...
    async def _refresh_dead_letter_depth(
        self,
        client: AsyncRedisClient,
        stream: str,
        dlq_stream: str,
    ) -> None: ...


class RedisDeadLetterMixin:
    """Move over-retried Redis Stream entries into a DLQ stream."""

    async def sweep_dead_letters(self: _RedisDeadLetterState, stream: str) -> int:
        """Move over-retried pending entries to ``<stream>.dlq``."""
        budget = self._dead_letter_after_retries
        if budget <= 0:
            return 0
        client = await self._get_client()
        try:
            groups = await client.xinfo_groups(stream)
        except Exception:  # pragma: no cover - stream may not exist
            return 0

        dlq_stream = f"{stream}.dlq"
        moved = 0
        for group in groups or []:
            group_name = group.get("name") if isinstance(group, dict) else None
            if not group_name:
                continue
            try:
                pending = await self._pending_entries_for_group(client, stream, group_name)
            except Exception as exc:  # pragma: no cover - advisory only
                log.debug(
                    "redis_streams.dead_letter_pending_fetch_failed",
                    stream=stream,
                    group=str(group_name),
                    error=str(exc),
                )
                continue
            for entry in pending:
                moved += await self._maybe_dead_letter_entry(
                    client=client,
                    stream=stream,
                    dlq_stream=dlq_stream,
                    group_name=str(group_name),
                    entry=entry,
                    budget=budget,
                )

        await self._refresh_dead_letter_depth(client, stream, dlq_stream)
        if moved:
            log.info(
                "redis_streams.dead_letter_moved",
                stream=stream,
                moved=moved,
            )
        return moved

    async def _pending_entries_for_group(
        self,
        client: AsyncRedisClient,
        stream: str,
        group_name: str,
    ) -> list[RedisPendingEntry]:
        cursor = "-"
        all_pending: list[RedisPendingEntry] = []
        while True:
            page = await client.xpending_range(
                name=stream,
                groupname=group_name,
                min=cursor,
                max="+",
                count=100,
            )
            if not page:
                break
            all_pending.extend(page)
            last_id = page[-1].get("message_id") if isinstance(page[-1], dict) else None
            if last_id is None or len(page) < 100:
                break
            cursor = last_id
        return all_pending

    async def _maybe_dead_letter_entry(
        self,
        *,
        client: AsyncRedisClient,
        stream: str,
        dlq_stream: str,
        group_name: str,
        entry: RedisPendingEntry,
        budget: int,
    ) -> int:
        message_id = entry.get("message_id")
        delivery_count = int(entry.get("times_delivered", 0))
        if not message_id or delivery_count <= budget:
            return 0

        try:
            rows = await client.xrange(stream, min=str(message_id), max=str(message_id), count=1)
        except Exception:  # pragma: no cover - advisory only
            return 0
        if not rows:
            await client.xack(stream, group_name, str(message_id))
            return 0

        _mid, fields = rows[0]
        try:
            await client.xadd(
                dlq_stream,
                {
                    **fields,
                    "orig_stream": stream,
                    "orig_group": group_name,
                    "orig_id": message_id,
                    "delivery_count": str(delivery_count),
                },
            )
            await client.xack(stream, group_name, str(message_id))
            record_dead_letter("redis_streams", stream, group_name)
            return 1
        except Exception as exc:  # pragma: no cover - connectivity
            log.warning(
                "redis_streams.dead_letter_failed",
                stream=stream,
                group=group_name,
                error=str(exc),
            )
            return 0

    async def _refresh_dead_letter_depth(
        self,
        client: AsyncRedisClient,
        stream: str,
        dlq_stream: str,
    ) -> None:
        try:
            depth = await client.xlen(dlq_stream)
            set_dead_letter_depth("redis_streams", stream, int(depth))
        except Exception:  # pragma: no cover - DLQ stream may not exist yet
            set_dead_letter_depth("redis_streams", stream, 0)
