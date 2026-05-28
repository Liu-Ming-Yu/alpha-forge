"""Broker health and paper-probe composition helpers."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.bootstrap.broker.paper_lifecycle import (
    ib_paper_lifecycle,
    paper_lifecycle_limit_price,
)
from quant_platform.bootstrap.broker.probe import (
    broker_gate_settings,
    classify_broker_probe_failure,
)
from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema
from quant_platform.core.domain.production import BrokerSmokeObservation

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping

    from quant_platform.config import PlatformSettings

__all__ = [
    "broker_gate_settings",
    "broker_health",
    "broker_smoke_from_report",
    "classify_broker_probe_failure",
    "ib_gateway_smoke",
    "ib_paper_lifecycle",
    "paper_lifecycle_limit_price",
    "sweep_dead_letters",
]


async def broker_health(settings: PlatformSettings) -> dict[str, object]:
    from quant_platform.bootstrap.session.public_api import create_paper_session
    from quant_platform.core.domain.production import BrokerHealthObservation
    from quant_platform.infrastructure.performance import build_performance_repository

    await verify_postgres_schema(settings)
    session = create_paper_session(settings, initial_cash=Decimal("0"))
    await session.broker.connect()
    try:
        health = await session.account_broker.health_check()
        if settings.storage.postgres_dsn:
            repo = build_performance_repository(settings.storage.postgres_dsn)
            await repo.save_broker_health(
                BrokerHealthObservation(
                    observed_at=datetime.now(tz=UTC),
                    status=health.status.value,
                    latency_ms=health.latency_ms,
                    last_heartbeat_at=health.last_heartbeat_at,
                    detail=health.detail,
                )
            )
        return {
            "status": health.status.value,
            "latency_ms": health.latency_ms,
            "last_heartbeat_at": str(health.last_heartbeat_at),
            "detail": health.detail,
            "kill_switch_active": session.execution_policy.kill_switch_active,
        }
    finally:
        await session.broker.disconnect()


async def ib_gateway_smoke(
    settings: PlatformSettings,
    contracts: Mapping[uuid.UUID, dict[str, object]],
) -> dict[str, object]:
    """Run a read-only IB Gateway smoke probe against the configured account."""

    from quant_platform.core.domain.production import (
        BrokerHealthObservation,
        BrokerSmokeObservation,
    )
    from quant_platform.infrastructure.performance import build_performance_repository
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    await verify_postgres_schema(settings)
    broker_settings = broker_gate_settings(settings)
    gateway = IBGatewayBrokerGateway(
        settings=broker_settings,
        instrument_contracts=dict(contracts),
    )
    started = time.monotonic()
    observed_at = datetime.now(tz=UTC)
    account_status = "skipped"
    positions_status = "skipped"
    open_orders_status = "skipped"
    status = "disconnected"
    detail = ""
    latency_ms = 0.0
    health = None
    account = None
    positions = []
    open_orders = []
    try:
        await gateway.connect()
        health = await gateway.health_check()
        status = health.status.value
        latency_ms = health.latency_ms
        account = await gateway.sync_account()
        account_status = "ok"
        positions = await gateway.sync_positions()
        positions_status = "ok"
        open_orders = await gateway.fetch_open_orders()
        open_orders_status = "ok"
        detail = health.detail
    except Exception as exc:
        latency_ms = max(0.0, (time.monotonic() - started) * 1000.0)
        detail = f"{classify_broker_probe_failure(exc)}: {exc}"
        status = "disconnected"
    finally:
        try:
            await gateway.disconnect()
        except Exception as exc:
            detail = f"{detail}; disconnect_error={exc}" if detail else f"disconnect_error={exc}"

    passed = (
        status == "connected"
        and account_status == "ok"
        and positions_status == "ok"
        and open_orders_status == "ok"
    )
    smoke = BrokerSmokeObservation(
        observed_at=observed_at,
        status=status,
        host=broker_settings.host,
        port=broker_settings.port,
        client_id=broker_settings.client_id,
        latency_ms=latency_ms,
        account_status=account_status,
        positions_status=positions_status,
        open_orders_status=open_orders_status,
        detail=detail,
    )
    if settings.storage.postgres_dsn:
        repo = build_performance_repository(settings.storage.postgres_dsn)
        await repo.save_broker_smoke(smoke)
        await repo.save_broker_health(
            BrokerHealthObservation(
                observed_at=observed_at,
                status=status,
                latency_ms=latency_ms,
                last_heartbeat_at=health.last_heartbeat_at if health is not None else None,
                detail=detail,
            )
        )
    return {
        "passed": passed,
        "status": status,
        "failure_type": None if passed else detail.split(":", 1)[0],
        "host": broker_settings.host,
        "port": broker_settings.port,
        "client_id": broker_settings.client_id,
        "latency_ms": latency_ms,
        "last_heartbeat_at": health.last_heartbeat_at if health is not None else None,
        "account_status": account_status,
        "positions_status": positions_status,
        "open_orders_status": open_orders_status,
        "account_as_of": account.as_of if account is not None else None,
        "net_asset_value": account.net_asset_value if account is not None else None,
        "settled_cash": account.settled_cash if account is not None else None,
        "positions": len(positions),
        "open_orders": len(open_orders),
        "paper_trading": broker_settings.paper_trading,
        "detail": detail,
        "observed_at": observed_at,
    }


async def sweep_dead_letters(settings: PlatformSettings, stream: str) -> tuple[int, int]:
    redis_url = settings.storage.redis_url
    if not redis_url:
        raise OperatorUsageError(
            "QP__STORAGE__REDIS_URL is not set; event-bus sweep has nothing to do."
        )
    from quant_platform.infrastructure.event_bus import RedisStreamsEventBus

    bus = RedisStreamsEventBus(
        redis_url=redis_url,
        stream_prefix=settings.storage.redis_stream_prefix,
        maxlen=settings.storage.redis_stream_maxlen,
        block_ms=settings.storage.redis_stream_block_ms,
        use_consumer_groups=settings.storage.redis_stream_use_consumer_groups,
        group_prefix=settings.storage.redis_stream_group_prefix,
        publish_dedupe_enabled=settings.storage.redis_stream_publish_dedupe_enabled,
        dedupe_ttl_seconds=settings.storage.redis_stream_dedupe_ttl_seconds,
        dead_letter_after_retries=settings.storage.redis_stream_dead_letter_after_retries,
    )
    moved = await bus.sweep_dead_letters(stream)
    client = await bus._get_client()  # noqa: SLF001 - intentional operational probe
    try:
        depth = int(await client.xlen(f"{stream}.dlq"))
    except Exception:
        depth = 0
    return moved, depth


def broker_smoke_from_report(report: dict[str, object]) -> BrokerSmokeObservation:
    from typing import cast

    return BrokerSmokeObservation(
        observed_at=cast("datetime", report["observed_at"]),
        status=str(report["status"]),
        host=str(report["host"]),
        port=_int_report_value(report, "port"),
        client_id=_int_report_value(report, "client_id"),
        latency_ms=_float_report_value(report, "latency_ms"),
        account_status=str(report["account_status"]),
        positions_status=str(report["positions_status"]),
        open_orders_status=str(report["open_orders_status"]),
        detail=str(report["detail"]),
    )


def _int_report_value(report: Mapping[str, object], key: str) -> int:
    value = report[key]
    if isinstance(value, (str, int, float)):
        return int(value)
    raise TypeError(f"{key} must be int-compatible, got {type(value).__name__}")


def _float_report_value(report: Mapping[str, object], key: str) -> float:
    value = report[key]
    if isinstance(value, (str, int, float)):
        return float(value)
    raise TypeError(f"{key} must be float-compatible, got {type(value).__name__}")
