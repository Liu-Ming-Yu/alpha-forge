"""Operator API dependency composition."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from quant_platform.application.operator_api.read_models import (
    CashLedgerViewPort,
    OperatorReadModelBuilder,
    ThrottleStateViewPort,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.infrastructure.event_bus import InMemoryAuditSink, create_event_bus
from quant_platform.infrastructure.repositories import (
    InMemoryOrderRepository,
    InMemoryPositionRepository,
)
from quant_platform.infrastructure.support.clock import WallClock
from quant_platform.services.execution_service.orders.order_throttle import OrderThrottle
from quant_platform.services.portfolio_service.cash_ledger import CashLedger
from quant_platform.services.portfolio_service.settlement_calendar import SettlementCalendar

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


@dataclass(frozen=True)
class OperatorApiRuntime:
    clock: Any
    cash_ledger: CashLedgerViewPort
    throttle: ThrottleStateViewPort
    audit_sink: Any
    v2_auth_repo: Any | None
    builder: OperatorReadModelBuilder
    position_repo: Any = None


def build_operator_api_runtime(
    settings: PlatformSettings,
    *,
    initial_cash: Decimal,
    cash_ledger: CashLedgerViewPort | None = None,
    throttle: ThrottleStateViewPort | None = None,
    order_repo: Any | None = None,
    position_repo: Any | None = None,
    performance_repo: Any | None = None,
    multi_engine_repo: Any | None = None,
    signal_contribution_repo: Any | None = None,
    event_bus: Any | None = None,
    audit_sink: Any | None = None,
    account_broker: Any | None = None,
) -> OperatorApiRuntime:
    clock = WallClock()
    cal = SettlementCalendar()
    snapshot_stub = _empty_snapshot(clock, initial_cash)
    ledger: CashLedgerViewPort = (
        cash_ledger
        if cash_ledger is not None
        else CashLedger(
            clock=clock,
            settlement_calendar=cal,
            initial_snapshot=snapshot_stub,
            settings=settings.cash,
        )
    )
    throttle_policy: ThrottleStateViewPort = (
        throttle if throttle is not None else OrderThrottle(clock, settings=settings.throttle)
    )
    events = event_bus or create_event_bus(
        backend=settings.storage.event_bus_backend,
        redis_url=settings.storage.redis_url,
        stream_prefix=settings.storage.redis_stream_prefix,
        stream_maxlen=settings.storage.redis_stream_maxlen,
        stream_block_ms=settings.storage.redis_stream_block_ms,
        stream_use_consumer_groups=settings.storage.redis_stream_use_consumer_groups,
        stream_group_prefix=settings.storage.redis_stream_group_prefix,
        stream_publish_dedupe_enabled=settings.storage.redis_stream_publish_dedupe_enabled,
        stream_dedupe_ttl_seconds=settings.storage.redis_stream_dedupe_ttl_seconds,
    )

    selected_order_repo = order_repo
    selected_position_repo = position_repo
    selected_audit_sink = audit_sink
    selected_multi_engine_repo = multi_engine_repo
    if selected_order_repo is None or selected_position_repo is None or selected_audit_sink is None:
        if settings.storage.postgres_dsn:
            from quant_platform.infrastructure.postgres.repositories import (
                PostgresAuditSink,
                PostgresOrderRepository,
                PostgresPositionRepository,
                create_pg_engine,
            )

            pg_engine = create_pg_engine(settings.storage.postgres_dsn)
            selected_order_repo = selected_order_repo or PostgresOrderRepository(pg_engine)
            selected_position_repo = selected_position_repo or PostgresPositionRepository(pg_engine)
            selected_audit_sink = selected_audit_sink or PostgresAuditSink(pg_engine)
        else:
            selected_order_repo = selected_order_repo or InMemoryOrderRepository()
            selected_position_repo = selected_position_repo or InMemoryPositionRepository()
            selected_audit_sink = selected_audit_sink or InMemoryAuditSink()
    if selected_multi_engine_repo is None and settings.storage.postgres_dsn:
        from quant_platform.infrastructure.repositories.multi_engine_governance import (
            build_multi_engine_governance_repository,
        )

        selected_multi_engine_repo = build_multi_engine_governance_repository(
            settings.storage.postgres_dsn
        )
    selected_signal_contribution_repo = signal_contribution_repo
    if selected_signal_contribution_repo is None and settings.storage.postgres_dsn:
        from quant_platform.infrastructure.repositories.signal_contributions import (
            build_signal_contribution_repository,
        )

        selected_signal_contribution_repo = build_signal_contribution_repository(
            settings.storage.postgres_dsn
        )
    selected_prediction_evidence_repo = performance_repo
    if selected_prediction_evidence_repo is None and settings.storage.postgres_dsn:
        from quant_platform.infrastructure.performance import build_performance_repository

        selected_prediction_evidence_repo = build_performance_repository(
            settings.storage.postgres_dsn
        )

    v2_auth_repo: Any | None = None
    if settings.v2.enabled and settings.storage.postgres_dsn:
        from quant_platform.infrastructure.v2.postgres import build_v2_repository_bundle

        v2_auth_repo = build_v2_repository_bundle(
            settings,
            require_postgres=True,
        ).production_evidence

    builder = OperatorReadModelBuilder(
        clock=clock,
        cash_ledger=ledger,
        throttle=throttle_policy,
        order_repo=selected_order_repo,
        position_repo=selected_position_repo,
        performance_repo=performance_repo,
        event_bus=events,
        account_broker=account_broker,
        multi_engine_repo=selected_multi_engine_repo,
        signal_contribution_repo=selected_signal_contribution_repo,
        prediction_evidence_repo=selected_prediction_evidence_repo,
    )
    return OperatorApiRuntime(
        clock=clock,
        cash_ledger=ledger,
        throttle=throttle_policy,
        audit_sink=selected_audit_sink,
        v2_auth_repo=v2_auth_repo,
        builder=builder,
        position_repo=selected_position_repo,
    )


def _empty_snapshot(clock: Any, cash: Decimal) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=clock.now(),
        settled_cash=cash,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=cash,
        net_asset_value=cash,
        positions=(),
    )
