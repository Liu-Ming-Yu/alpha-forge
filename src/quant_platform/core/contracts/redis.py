"""Typed async Redis client contracts for adapter IO boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

RedisHash: TypeAlias = dict[str, str]
RedisStreamMessage: TypeAlias = tuple[str, RedisHash]
RedisStreamResponse: TypeAlias = list[tuple[str, list[RedisStreamMessage]]]
RedisGroupInfo: TypeAlias = dict[str, Any]
RedisPendingEntry: TypeAlias = dict[str, Any]
RedisSortedSetRow: TypeAlias = tuple[str, float]


class AsyncRedisClient(Protocol):
    """Subset of redis.asyncio used by platform adapters."""

    async def aclose(self) -> None: ...
    async def delete(self, *keys: str) -> int: ...
    async def eval(self, script: str, numkeys: int, *keys_and_args: object) -> object: ...
    async def expire(self, name: str, time: int) -> bool: ...
    async def get(self, name: str) -> str | None: ...
    async def ping(self) -> bool: ...
    def scan_iter(self, match: str, count: int | None = None) -> AsyncIterator[str]: ...
    async def set(self, name: str, value: object, *args: object, **kwargs: object) -> object: ...
    async def xack(self, name: str, groupname: str, *ids: str) -> int: ...
    async def xadd(
        self,
        name: str,
        fields: Mapping[str, object],
        *args: object,
        **kwargs: object,
    ) -> str: ...
    async def xgroup_create(
        self,
        name: str,
        groupname: str,
        *args: object,
        **kwargs: object,
    ) -> object: ...
    async def xinfo_groups(self, name: str) -> list[RedisGroupInfo]: ...
    async def xlen(self, name: str) -> int: ...
    async def xpending(self, name: str, groupname: str) -> dict[str, Any] | list[Any]: ...
    async def xpending_range(
        self,
        name: str,
        groupname: str,
        min: str,
        max: str,
        count: int,
    ) -> list[RedisPendingEntry]: ...
    async def xrange(
        self,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> list[RedisStreamMessage]: ...
    async def xread(
        self,
        streams: dict[str, str],
        *args: object,
        **kwargs: object,
    ) -> RedisStreamResponse: ...
    async def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        *args: object,
        **kwargs: object,
    ) -> RedisStreamResponse: ...
    async def xrevrange(
        self,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> list[RedisStreamMessage]: ...
    async def xtrim(self, name: str, *args: object, **kwargs: object) -> int: ...
    async def zadd(self, name: str, mapping: dict[str, float]) -> int: ...
    async def zcard(self, name: str) -> int: ...
    async def zrange(
        self,
        name: str,
        start: int,
        end: int,
        *args: object,
        **kwargs: object,
    ) -> list[RedisSortedSetRow]: ...
    async def zremrangebyscore(self, name: str, min: str, max: str) -> int: ...
