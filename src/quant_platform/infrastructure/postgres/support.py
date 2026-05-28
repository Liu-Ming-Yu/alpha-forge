"""Shared PostgreSQL adapter support."""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, TypeVar, cast

import structlog
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

log = structlog.get_logger(__name__)

# PostgreSQL SQLSTATE class 08: connection exceptions, always transient.
_TRANSIENT_SQLSTATES: frozenset[str] = frozenset(
    {"08000", "08001", "08003", "08004", "08006", "08007", "08P01"}
)
_P = ParamSpec("_P")
_T = TypeVar("_T")


class PoolMetrics(Protocol):
    def checkedout(self) -> int: ...

    def checkedin(self) -> int: ...

    def overflow(self) -> int: ...


def retry_transient(
    attempts: int = 3,
    base_delay: float = 0.5,
) -> Callable[
    [Callable[_P, Coroutine[Any, Any, _T]]],
    Callable[_P, Coroutine[Any, Any, _T]],
]:
    """Retry an async repository method on transient PostgreSQL connection errors."""

    def decorator(
        fn: Callable[_P, Coroutine[Any, Any, _T]],
    ) -> Callable[_P, Coroutine[Any, Any, _T]]:
        @functools.wraps(fn)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            for attempt in range(attempts):
                try:
                    return await fn(*args, **kwargs)
                except OperationalError as exc:
                    orig = getattr(exc, "orig", None)
                    sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
                    if sqlstate not in _TRANSIENT_SQLSTATES or attempt >= attempts - 1:
                        raise
                    delay = base_delay * (2**attempt)
                    log.warning(
                        "pg.transient_error.retry",
                        fn=fn.__qualname__,
                        attempt=attempt + 1,
                        sqlstate=sqlstate,
                        retry_delay_s=delay,
                    )
                    await asyncio.sleep(delay)
            raise RuntimeError("unreachable retry loop exit")

        return wrapper

    return decorator


@functools.lru_cache(maxsize=8)
def create_pg_engine(
    dsn: str,
    *,
    pool_size: int = 10,
    max_overflow: int = 10,
    statement_timeout_ms: int = 30_000,
) -> AsyncEngine:
    """Create or return a cached async SQLAlchemy engine for the given DSN."""
    from sqlalchemy import event

    from quant_platform.infrastructure.metrics import set_db_pool_utilization

    async_dsn = dsn
    if async_dsn.startswith("postgresql://"):
        async_dsn = async_dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    connect_args: dict[str, object] = {"connect_timeout": 30}
    if statement_timeout_ms > 0:
        connect_args["options"] = f"-c statement_timeout={statement_timeout_ms}"
    engine = create_async_engine(
        async_dsn,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
        execution_options={"isolation_level": "SERIALIZABLE"},
        connect_args=connect_args,
    )

    def _update_pool_metrics() -> None:
        pool = cast("PoolMetrics", engine.sync_engine.pool)
        set_db_pool_utilization(
            checked_out=pool.checkedout(),
            idle=pool.checkedin(),
            overflow=pool.overflow(),
        )

    @event.listens_for(engine.sync_engine, "checkout")
    def _on_checkout(dbapi_conn: object, conn_record: object, conn_proxy: object) -> None:
        _update_pool_metrics()

    @event.listens_for(engine.sync_engine, "checkin")
    def _on_checkin(dbapi_conn: object, conn_record: object) -> None:
        _update_pool_metrics()

    return engine


_retry_transient = retry_transient
