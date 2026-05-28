"""Strategy-cycle lifecycle helpers for locks and NAV snapshots."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.production import NavSnapshot
from quant_platform.infrastructure.support.distributed_lock import create_distributed_lock

if TYPE_CHECKING:
    from quant_platform.application.runtime.state import Session

log = structlog.get_logger(__name__)

DistributedLockFactory = Callable[..., AbstractAsyncContextManager[object]]


async def record_nav_snapshot(session: Session, strategy_run_id: uuid.UUID) -> None:
    """Persist a best-effort NAV snapshot for operator lifecycle metrics."""
    try:
        account = await session.account_broker.sync_account()
        gross_exposure = sum(
            (position.market_value for position in account.positions),
            Decimal("0"),
        )
        await session.performance_repo.save_nav_snapshot(
            NavSnapshot(
                snapshot_id=uuid.uuid4(),
                strategy_run_id=strategy_run_id,
                as_of=account.as_of,
                net_asset_value=account.net_asset_value,
                gross_exposure=gross_exposure,
                cash=account.settled_cash,
                source=account.source,
            )
        )
    except Exception as exc:  # pragma: no cover - observability best-effort
        log.warning("strategy_cycle.nav_snapshot_failed", error=str(exc))


@asynccontextmanager
async def strategy_cycle_lock(
    session: Session,
    strategy_run_id: uuid.UUID,
    *,
    lock_factory: DistributedLockFactory = create_distributed_lock,
) -> AsyncIterator[object]:
    """Acquire a distributed lock for mutating strategy-cycle state."""
    lock_name = f"strategy_cycle:{strategy_run_id}"
    lock = lock_factory(
        session.settings.storage.redis_url,
        lock_name,
        ttl_seconds=session.settings.storage.distributed_lock_ttl_seconds,
        acquire_timeout_seconds=session.settings.storage.distributed_lock_acquire_timeout_seconds,
        renew_interval_seconds=session.settings.storage.distributed_lock_renew_interval_seconds,
    )
    try:
        async with lock:
            yield lock
    finally:
        # Release the underlying Redis client so the connection pool
        # doesn't accumulate one fresh client per cycle. The lock itself
        # was already released by ``async with``.
        aclose = getattr(lock, "aclose", None)
        if aclose is not None:
            await aclose()
