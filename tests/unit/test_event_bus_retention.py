"""Tests for the XTRIM sweeper (Phase 4.2)."""

from __future__ import annotations

import pytest

from quant_platform.services.data_service.maintenance.event_bus_retention import (
    EventBusRetentionWorker,
)


class _FakeRedis:
    """Minimal ``XTRIM``-compatible fake."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_on: set[str] = set()
        self.groups: dict[str, list[dict[str, object]]] = {}
        self.pending: dict[tuple[str, str], list[dict[str, str]]] = {}

    async def xtrim(self, stream: str, *, minid: str, approximate: bool) -> int:
        if stream in self.fail_on:
            raise RuntimeError(f"xtrim failed for {stream!r}")
        self.calls.append((stream, minid))
        return 42

    async def xinfo_groups(self, stream: str) -> list[dict[str, object]]:
        return self.groups.get(stream, [])

    async def xpending_range(
        self,
        *,
        name: str,
        groupname: str,
        min: str,
        max: str,
        count: int,
    ) -> list[dict[str, str]]:
        _ = min, max, count
        return self.pending.get((name, groupname), [])


@pytest.mark.asyncio
async def test_sweep_once_trims_all_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRedis()
    worker = EventBusRetentionWorker(
        redis_url="redis://x",
        stream_keys=["qp:events:FillReceived", "qp:events:OrderAcknowledged"],
        retention_ms=60_000,
    )
    worker._redis = fake  # noqa: SLF001 - test injection
    summary = await worker.sweep_once()

    assert set(summary.keys()) == {
        "qp:events:FillReceived",
        "qp:events:OrderAcknowledged",
    }
    assert all(v == 42 for v in summary.values())
    assert len(fake.calls) == 2
    assert all(call[1].endswith("-0") for call in fake.calls)


@pytest.mark.asyncio
async def test_sweep_once_disabled_when_retention_zero() -> None:
    worker = EventBusRetentionWorker(
        redis_url="redis://x",
        stream_keys=["qp:events:FillReceived"],
        retention_ms=0,
    )
    summary = await worker.sweep_once()
    assert summary == {}


@pytest.mark.asyncio
async def test_sweep_once_reports_negative_on_error() -> None:
    fake = _FakeRedis()
    fake.fail_on = {"qp:events:broken"}
    worker = EventBusRetentionWorker(
        redis_url="redis://x",
        stream_keys=["qp:events:broken", "qp:events:ok"],
        retention_ms=60_000,
    )
    worker._redis = fake  # noqa: SLF001

    summary = await worker.sweep_once()
    assert summary["qp:events:broken"] == -1
    assert summary["qp:events:ok"] == 42


@pytest.mark.asyncio
async def test_sweep_once_preserves_oldest_pending_entry() -> None:
    fake = _FakeRedis()
    stream = "qp:events:OrderSubmitted"
    fake.groups[stream] = [{"name": "g1", "pending": 2}]
    fake.pending[(stream, "g1")] = [{"message_id": "1000-0"}]
    worker = EventBusRetentionWorker(
        redis_url="redis://x",
        stream_keys=[stream],
        retention_ms=60_000,
    )
    worker._redis = fake  # noqa: SLF001

    summary = await worker.sweep_once()

    assert summary[stream] == 42
    assert fake.calls == [(stream, "1000-0")]


def test_negative_retention_rejected() -> None:
    with pytest.raises(ValueError):
        EventBusRetentionWorker(
            redis_url="redis://x",
            stream_keys=["s"],
            retention_ms=-1,
        )
