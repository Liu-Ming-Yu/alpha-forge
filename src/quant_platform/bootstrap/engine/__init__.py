"""Engine CLI/runtime composition helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from quant_platform.bootstrap.engine.loop import (
    EngineLoopConfig,
    EngineLoopSummary,
    run_engine_loop,
)
from quant_platform.bootstrap.engine.multi import (
    latest_contract_market_prices,
    load_budgets,
    run_multi_engine_v2,
)
from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema
from quant_platform.bootstrap.session.public_api import create_paper_session
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.engines.session.public_api import run_strategy_cycle

if TYPE_CHECKING:
    from decimal import Decimal

    from quant_platform.application.runtime.state import Session
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)


async def run_cycle_once(
    settings: PlatformSettings,
    *,
    initial_cash: Decimal,
    feature_data: dict[uuid.UUID, dict[str, float]] | None = None,
) -> Any:
    await verify_postgres_schema(settings)
    session = create_paper_session(settings, initial_cash=initial_cash)
    await session.broker.connect()
    try:
        await _startup_health_check(session)
        strategy_run = _make_strategy_run(settings)
        result = await run_strategy_cycle(
            session=session,
            feature_data=feature_data or {},
            strategy_run=strategy_run,
        )
    finally:
        await session.broker.disconnect()
    return result


async def supervise_engine(
    settings: PlatformSettings,
    *,
    initial_cash: Decimal,
    interval_seconds: float,
    mode: str = "paper",
    max_cycles: int | None = None,
    contracts_file: str | None = None,
    engine_name: str = "cross_sectional_equity",
    execution_backend: str = "simulated",
) -> EngineLoopSummary:
    if mode == "live":
        raise ValueError(
            "supervise does not support live mode; use bounded run-engine/V2 live flow"
        )
    return await run_engine_loop(
        settings,
        EngineLoopConfig(
            engine_name=engine_name,
            mode=mode,
            execution_backend=execution_backend,
            initial_cash=initial_cash,
            contracts_file=contracts_file,
            interval_seconds=interval_seconds,
            max_cycles=max_cycles,
            install_signal_handlers=True,
        ),
    )


def _make_strategy_run(settings: PlatformSettings) -> StrategyRun:
    now = datetime.now(tz=UTC)
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="cli_cycle",
        strategy_version="0.1.0",
        run_type=RunType.PAPER if settings.broker.paper_trading else RunType.LIVE,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=now,
        started_at=now,
    )


async def _startup_health_check(session: Session) -> None:
    from quant_platform.core.contracts.common import BrokerHealthStatus

    health = await session.account_broker.health_check()
    if health.status != BrokerHealthStatus.CONNECTED:
        raise RuntimeError(
            f"startup health check failed: broker status={health.status.value} "
            f"(detail={health.detail})"
        )
    log.info("startup.broker_healthy", latency_ms=health.latency_ms)

    redis_url = session.settings.storage.redis_url
    if redis_url:
        try:
            from quant_platform.bootstrap.persistence.redis_factory import create_async_redis_client

            client = create_async_redis_client(redis_url, socket_timeout=5.0)
            ok = await client.ping()
            await client.aclose()
            if not ok:
                raise RuntimeError("startup health check failed: Redis PING returned False")
            log.info("startup.redis_healthy")
        except Exception as exc:
            raise RuntimeError(f"startup health check failed: Redis unreachable: {exc}") from exc


__all__ = [
    "latest_contract_market_prices",
    "load_budgets",
    "run_cycle_once",
    "run_engine_loop",
    "run_multi_engine_v2",
    "supervise_engine",
]
