"""Live-mode runtime helpers for engine sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import BrokerSettings, PlatformSettings
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot

log = structlog.get_logger(__name__)


def assert_v2_is_only_live_submitter(settings: PlatformSettings) -> None:
    """Block the single-engine live submit path when V2 owns live orders."""
    v2 = settings.v2
    if not (v2.enabled and v2.account_orchestrator_enabled):
        return
    raise RuntimeError(
        "V2 account orchestrator is enabled "
        "(QP__V2__ENABLED=true and QP__V2__ACCOUNT_ORCHESTRATOR_ENABLED=true). "
        "The single-engine EngineRunner live path is blocked because "
        "every live order must be merged and submitted by "
        "AccountExecutionOrchestrator.  Use 'python -m quant_platform run-multi-engine' "
        "or leave `run-engine --mode live` on its automatic V2 delegation path."
    )


async def bootstrap_live_snapshot(
    *,
    broker_settings: BrokerSettings,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> AccountSnapshot:
    """Connect once to IB Gateway and fetch the authoritative account snapshot."""
    return await _bootstrap_ib_snapshot(
        broker_settings=broker_settings,
        instrument_contracts=instrument_contracts,
    )


async def bootstrap_ib_paper_snapshot(
    *,
    broker_settings: BrokerSettings,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> AccountSnapshot:
    """Connect once to paper TWS/Gateway and fetch the authoritative snapshot."""
    return await _bootstrap_ib_snapshot(
        broker_settings=broker_settings,
        instrument_contracts=instrument_contracts,
    )


async def _bootstrap_ib_snapshot(
    *,
    broker_settings: BrokerSettings,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> AccountSnapshot:
    """Fetch one broker snapshot through the IB adapter."""
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    bootstrap_gateway = IBGatewayBrokerGateway(
        settings=broker_settings,
        instrument_contracts=instrument_contracts,
    )
    await bootstrap_gateway.connect()
    try:
        return await bootstrap_gateway.sync_account()
    finally:
        await bootstrap_gateway.disconnect()
