"""Read-only operator HTTP API.

Thin FastAPI wrapper around ``OperatorReadModelBuilder``.  Serves session
health, cash status, order blotter, and paper-gate metrics as JSON.

Start with::

    python -m quant_platform serve-api --initial-cash 50000

or directly::

    uvicorn quant_platform.views.operator_api.app:create_app --factory --port 8000

Requires the ``api`` extra: ``pip install -e ".[api]"``
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import structlog

from quant_platform.bootstrap.operator_api.dependencies import build_operator_api_runtime
from quant_platform.bootstrap.operator_api.research_queries import (
    build_operator_research_query_service,
)
from quant_platform.config import PlatformSettings, configure_logging
from quant_platform.views.operator_api.lifespan import build_operator_api_lifespan
from quant_platform.views.operator_api.middleware import install_operator_api_middlewares
from quant_platform.views.operator_api.security import resolve_operator_api_key

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Protocol

    from fastapi import FastAPI

    from quant_platform.application.operator_api.read_model_types import (
        CashLedgerViewPort,
        ThrottleStateViewPort,
    )
    from quant_platform.application.runtime.state import Session
    from quant_platform.core.contracts import (
        AuditSink,
        BrokerSessionGateway,
        EventBus,
        MultiEngineGovernanceRepository,
        OrderRepository,
        PerformanceRepository,
        PositionRepository,
        SignalContributionRepository,
    )

    class OperatorKillSwitchStore(Protocol):
        async def get(self) -> object: ...
        async def clear(self, *, operator_id: str, as_of: datetime) -> None: ...


_log = structlog.get_logger(__name__)


def create_app(
    settings: PlatformSettings | None = None,
    initial_cash: Decimal = Decimal("50000"),
    *,
    cash_ledger: CashLedgerViewPort | None = None,
    throttle: ThrottleStateViewPort | None = None,
    order_repo: OrderRepository | None = None,
    position_repo: PositionRepository | None = None,
    performance_repo: PerformanceRepository | None = None,
    multi_engine_repo: MultiEngineGovernanceRepository | None = None,
    signal_contribution_repo: SignalContributionRepository | None = None,
    event_bus: EventBus | None = None,
    audit_sink: AuditSink | None = None,
    account_broker: BrokerSessionGateway | None = None,
    kill_switch_store: OperatorKillSwitchStore | None = None,
) -> FastAPI:
    """Factory that returns a configured FastAPI application.

    Designed for use with ``uvicorn --factory`` or programmatic startup.
    """
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the operator API. Install with: pip install -e '.[api]'"
        ) from exc

    settings = settings or PlatformSettings()
    configure_logging(settings.logging)

    operator_api_key = resolve_operator_api_key(settings)

    runtime = build_operator_api_runtime(
        settings,
        initial_cash=initial_cash,
        cash_ledger=cash_ledger,
        throttle=throttle,
        order_repo=order_repo,
        position_repo=position_repo,
        performance_repo=performance_repo,
        multi_engine_repo=multi_engine_repo,
        signal_contribution_repo=signal_contribution_repo,
        event_bus=event_bus,
        audit_sink=audit_sink,
        account_broker=account_broker,
    )
    clock = runtime.clock
    ledger = runtime.cash_ledger
    throttle_policy = runtime.throttle
    selected_audit_sink = runtime.audit_sink
    v2_auth_repo = runtime.v2_auth_repo
    builder = runtime.builder

    # Mutable container shared by lifespan, signal handler, and health routes.
    # Using a list avoids the `nonlocal` dance across nested closures.
    _shutting_down: list[bool] = [False]

    async def _hydrate_ledger_from_latest_snapshot() -> None:
        # Seed the read-model cash ledger from the latest broker-authoritative
        # account snapshot (written by reconciliation) so the console shows the
        # REAL account, not the synthetic --initial-cash stub. Falls back to the
        # stub when no snapshot exists yet.
        repo = runtime.position_repo
        reset = getattr(ledger, "reset_from_snapshot", None)
        if repo is None or reset is None:
            return
        snapshot = await repo.get_latest_snapshot()
        if snapshot is None:
            return
        reset(snapshot)
        _log.info(
            "operator_api.ledger_hydrated",
            settled_cash=str(snapshot.settled_cash),
            net_asset_value=str(snapshot.net_asset_value),
            positions=len(snapshot.positions),
        )

    app = FastAPI(
        title="Quant Platform Operator API",
        version="0.1.0",
        description="Read-only operator views for session health, cash, and orders.",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=build_operator_api_lifespan(
            _shutting_down, on_startup=_hydrate_ledger_from_latest_snapshot
        ),
    )

    # --- CORS lockdown ------------------------------------------------------
    # Default is empty string -> deny all cross-origin requests.  Operators can
    # set ``QP__API__CORS_ALLOW_ORIGINS`` to a comma-separated list of origins.
    cors_raw = settings.api.cors_allow_origins.strip()
    if cors_raw:
        from fastapi.middleware.cors import CORSMiddleware

        origins = [origin.strip() for origin in cors_raw.split(",") if origin.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

    install_operator_api_middlewares(
        app,
        settings=settings,
        clock=clock,
        v2_auth_repo=v2_auth_repo,
        operator_api_key=operator_api_key,
    )

    from quant_platform.views.operator_api.routers import (
        OperatorApiRouteContext,
        register_backtest_routes,
        register_broker_routes,
        register_cash_orders_audit_routes,
        register_command_routes,
        register_console_config_routes,
        register_dashboard_data_routes,
        register_dashboard_routes,
        register_health_routes,
        register_research_evidence_routes,
        register_strategy_state_routes,
    )
    from quant_platform.views.operator_api.static import mount_operator_console

    protected_dependencies: list[Any] = []
    route_context = OperatorApiRouteContext(
        settings=settings,
        clock=clock,
        ledger=ledger,
        throttle_policy=throttle_policy,
        selected_audit_sink=selected_audit_sink,
        v2_auth_repo=v2_auth_repo,
        builder=builder,
        kill_switch_store=kill_switch_store,
        operator_api_key=operator_api_key,
        research_queries=build_operator_research_query_service(settings),
        shutting_down=_shutting_down,
        protected_dependencies=protected_dependencies,
    )

    register_health_routes(app, route_context)
    register_cash_orders_audit_routes(app, route_context)
    register_strategy_state_routes(app, route_context)
    register_research_evidence_routes(app, route_context)
    register_dashboard_routes(app, route_context)
    register_console_config_routes(app, route_context)
    register_command_routes(app, route_context)
    register_backtest_routes(app, route_context)
    register_broker_routes(app, route_context)
    register_dashboard_data_routes(app, route_context)

    # Serve the built console SPA last so any JSON API path always wins.
    mount_operator_console(app, settings)

    return app


def create_app_from_session(session: Session) -> FastAPI:
    """Create the operator API bound to an existing Session's live dependencies."""
    if not _looks_like_order_throttle(session.execution_policy):
        raise TypeError(
            "create_app_from_session requires Session.execution_policy to be OrderThrottle"
        )
    return create_app(
        settings=session.settings,
        initial_cash=session.cash_engine.settled_cash,
        cash_ledger=cast("CashLedgerViewPort", session.cash_engine),
        throttle=cast("ThrottleStateViewPort", session.execution_policy),
        order_repo=session.order_repo,
        position_repo=session.position_repo,
        performance_repo=getattr(session, "performance_repo", None),
        multi_engine_repo=getattr(session, "multi_engine_repo", None),
        event_bus=session.event_bus,
        audit_sink=getattr(session, "audit_sink", None),
        account_broker=getattr(session, "account_broker", None),
        signal_contribution_repo=getattr(session, "signal_contribution_repo", None),
        kill_switch_store=getattr(session, "kill_switch_store", None),
    )


def _looks_like_order_throttle(value: object) -> bool:
    return all(
        hasattr(value, name)
        for name in (
            "kill_switch_active",
            "total_submitted",
            "tokens_available",
            "clear_kill_switch",
        )
    )
