"""Typed Redis client factory for infrastructure adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from quant_platform.core.contracts.redis import AsyncRedisClient


def create_async_redis_client(redis_url: str, **kwargs: object) -> AsyncRedisClient:
    import redis.asyncio as aioredis

    from_url = cast("Any", vars(aioredis)["from_url"])
    return cast("AsyncRedisClient", from_url(redis_url, **kwargs))
