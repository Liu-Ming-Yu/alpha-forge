"""Unit tests for distributed lock renewal behavior."""

from __future__ import annotations

import asyncio

import pytest

from quant_platform.infrastructure.support.distributed_lock import (
    DistributedLock,
    NullLock,
    create_distributed_lock,
)


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        _ = ex
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def eval(self, script: str, _: int, key: str, holder: str, *args: object) -> int:
        if "expire" in script:
            _ = args
            return 1 if self.values.get(key) == holder else 0
        return 1 if self.values.pop(key, None) == holder else 0


@pytest.mark.asyncio
async def test_lock_lease_lost_when_owner_changes() -> None:
    fake = _FakeRedis()
    lock = DistributedLock(
        redis_url="redis://fake",
        name="test-lock",
        ttl_seconds=3,
        renew_interval_seconds=0.01,
    )
    lock._client = fake  # type: ignore[attr-defined]

    assert await lock.acquire()
    await asyncio.sleep(0.03)
    assert not lock.lease_lost

    fake.values[lock._name] = "different-holder"  # type: ignore[attr-defined]
    await asyncio.sleep(0.03)
    assert lock.lease_lost
    await lock.release()


def test_factory_returns_null_lock_without_redis() -> None:
    lock = create_distributed_lock("", "x")
    assert isinstance(lock, NullLock)
