"""Live broker and account-gateway wiring for session entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.production import ProductionProfile
from quant_platform.core.exceptions import InstrumentMappingError
from quant_platform.services.execution_service.gateways.client_portal_gateway import (
    ClientPortalBrokerGateway,
)
from quant_platform.services.execution_service.stores.pacing_store import build_pacing_store
from quant_platform.services.governance_service.preflight import (
    assert_preflight_passed,
    evaluate_preflight,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.config_broker import BrokerSettings
    from quant_platform.core.contracts import BrokerGateway, BrokerSessionGateway, Clock
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

IB_TWS_PAPER_PORT = 7497
IB_GATEWAY_PAPER_PORT = 4002
IB_PAPER_PORTS = frozenset({IB_TWS_PAPER_PORT, IB_GATEWAY_PAPER_PORT})


def _load_ib_gateway_type() -> type[IBGatewayBrokerGateway]:
    try:
        from quant_platform.services.execution_service.gateways.broker_gateway import (
            IBGatewayBrokerGateway,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "ibapi":
            raise RuntimeError(
                "IBKR live broker support requires the optional 'ibapi' package. "
                "Install ibapi or run paper/offline verification without live broker wiring."
            ) from exc
        raise
    return IBGatewayBrokerGateway


def validate_live_contracts(
    settings: PlatformSettings,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    *,
    context: str = "create_live_session",
) -> None:
    """Validate live-session instrument mappings before gateway construction."""
    if settings.broker.primary_broker_path == "tws" and not instrument_contracts:
        raise ValueError(
            f"{context}: instrument_contracts must not be empty for "
            "the 'tws' broker path.  At minimum one instrument mapping is required "
            "for order routing.  Provide a dict of {instrument_id: contract_spec}."
        )

    con_ids = [
        spec.get("con_id")
        for spec in instrument_contracts.values()
        if spec.get("con_id") is not None
    ]
    if len(con_ids) != len(set(con_ids)):
        raise ValueError(
            f"{context}: duplicate con_id detected in instrument_contracts. "
            "Each instrument must map to a unique IB contract ID."
        )


def validate_ib_paper_execution(
    settings: PlatformSettings,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> None:
    """Fail closed when IB-backed paper execution is not pointed at paper TWS."""
    validate_live_contracts(
        settings,
        instrument_contracts,
        context="create_ib_paper_session",
    )
    missing_con_id = []
    for instrument_id, spec in instrument_contracts.items():
        con_id = spec.get("con_id")
        if not isinstance(con_id, int) or con_id <= 0:
            missing_con_id.append(str(instrument_id))
    if missing_con_id:
        raise ValueError(
            "IB paper execution requires every instrument contract to include a "
            f"positive con_id; missing={len(missing_con_id)}."
        )
    broker = settings.broker
    if not broker.paper_trading:
        raise ValueError(
            "IB paper execution requires QP__BROKER__PAPER_TRADING=true; "
            "refusing to use a live broker configuration."
        )
    if broker.port not in IB_PAPER_PORTS:
        raise ValueError(
            "IB paper execution requires a paper TWS/Gateway port "
            f"({IB_TWS_PAPER_PORT} or {IB_GATEWAY_PAPER_PORT}); got {broker.port}."
        )
    account_id = broker.account_id.strip().upper()
    if account_id and not account_id.startswith("DU"):
        raise ValueError(
            "IB paper execution requires a DU paper account_id when configured; "
            f"got {broker.account_id!r}."
        )


def run_live_preflight(
    settings: PlatformSettings,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> None:
    """Run live preflight when dev defaults are disabled."""
    if settings.allow_dev_defaults:
        return
    preflight_report = evaluate_preflight(
        settings,
        profile=ProductionProfile.LIVE,
        instrument_contracts={k: v for k, v in instrument_contracts.items()},
    )
    assert_preflight_passed(preflight_report, ProductionProfile.LIVE)


def build_live_broker_gateways(
    *,
    settings: PlatformSettings,
    initial_snapshot: AccountSnapshot,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    clock: Clock,
) -> tuple[BrokerGateway, BrokerSessionGateway]:
    """Build the trading gateway and account gateway for a live session."""
    validate_live_contracts(settings, instrument_contracts)
    run_live_preflight(settings, instrument_contracts)
    return _build_ib_broker_gateways(
        broker_settings=settings.broker,
        initial_snapshot=initial_snapshot,
        instrument_contracts=instrument_contracts,
        clock=clock,
        redis_url=settings.storage.redis_url or None,
        mapping_error_context="create_live_session",
    )


def build_ib_paper_broker_gateways(
    *,
    settings: PlatformSettings,
    initial_snapshot: AccountSnapshot,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    clock: Clock,
) -> tuple[BrokerGateway, BrokerSessionGateway]:
    """Build the trading gateway and account gateway for an IB paper session."""
    validate_ib_paper_execution(settings, instrument_contracts)
    return _build_ib_broker_gateways(
        broker_settings=settings.broker,
        initial_snapshot=initial_snapshot,
        instrument_contracts=instrument_contracts,
        clock=clock,
        redis_url=settings.storage.redis_url or None,
        mapping_error_context="create_ib_paper_session",
    )


def _build_ib_broker_gateways(
    *,
    broker_settings: BrokerSettings,
    initial_snapshot: AccountSnapshot,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    clock: Clock,
    redis_url: str | None,
    mapping_error_context: str,
) -> tuple[BrokerGateway, BrokerSessionGateway]:
    """Construct IB-backed broker adapters after profile-specific validation."""
    pacing_store = build_pacing_store(
        redis_url=redis_url,
        client_id=broker_settings.client_id,
        window_seconds=broker_settings.historical_bar_pacing_window_seconds,
    )
    ib_gateway_type = _load_ib_gateway_type()
    trading_broker = ib_gateway_type(
        settings=broker_settings,
        instrument_contracts=instrument_contracts,
        pacing_store=pacing_store,
    )

    mapping_warnings = trading_broker.validate_instrument_mappings()
    if mapping_warnings:
        # Keep the user-facing exception message free of con_id and symbol
        # detail (it can flow into operator alerts and incident channels);
        # emit the full structured payload to logs only.
        mapping_log = structlog.get_logger("quant_platform.bootstrap.broker.live_broker_wiring")
        mapping_log.error(
            "live_broker_wiring.instrument_mapping_failed",
            count=len(mapping_warnings),
            warnings=list(mapping_warnings),
        )
        raise InstrumentMappingError(
            f"{mapping_error_context}: {len(mapping_warnings)} instrument(s) lack a valid "
            "con_id for position/fill reverse lookup. Live sessions require complete "
            "mappings to prevent reconciliation blind spots. See structured log "
            "'live_broker_wiring.instrument_mapping_failed' for the per-instrument detail."
        )

    if broker_settings.primary_broker_path == "tws":
        return trading_broker, trading_broker

    account_broker: BrokerSessionGateway = ClientPortalBrokerGateway(
        clock=clock,
        initial_snapshot=initial_snapshot,
        upstream_session_gateway=trading_broker,
    )
    return trading_broker, account_broker
