"""Broker operator adapters."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.application.operator.cli_inputs import load_instrument_contracts
from quant_platform.bootstrap.operator_adapters.common import contract_reference_prices

if TYPE_CHECKING:
    from quant_platform.application.operator.requests import (
        BrokerContractsRequest,
        EventBusSweepRequest,
        PaperLifecycleRequest,
        PassiveRepriceRequest,
    )
    from quant_platform.config import PlatformSettings
    from quant_platform.services.execution_service.passive_reprice.passive_reprice_models import (
        PassiveRepriceDecision,
    )


class BrokerAdapters:
    """Concrete broker adapters backed by execution bootstrap helpers."""

    def __init__(self, settings: PlatformSettings) -> None:
        self._settings = settings

    async def gateway_smoke(self, request: BrokerContractsRequest) -> dict[str, object]:
        from quant_platform.bootstrap.broker import ib_gateway_smoke

        return await ib_gateway_smoke(
            self._settings,
            load_instrument_contracts(request.contracts_file),
        )

    async def paper_lifecycle(self, request: PaperLifecycleRequest) -> dict[str, object]:
        from quant_platform.bootstrap.broker import ib_paper_lifecycle

        contracts = load_instrument_contracts(request.contracts_file)
        if request.instrument_id not in contracts:
            raise ValueError(
                f"instrument_id {request.instrument_id} is not present in {request.contracts_file}"
            )
        return await ib_paper_lifecycle(
            self._settings,
            contracts=contracts,
            instrument_id=request.instrument_id,
            max_notional_usd=request.max_notional_usd,
            max_allowed_notional=Decimal("100"),
        )

    async def passive_reprice(self, request: PassiveRepriceRequest) -> dict[str, object]:
        from quant_platform.bootstrap.session.public_factory import (
            create_live_session_impl,
            create_paper_session_impl,
        )
        from quant_platform.engines.runtime.live import bootstrap_live_snapshot
        from quant_platform.engines.session.passive_reprice import run_passive_reprice_once

        contracts = load_instrument_contracts(request.contracts_file)
        if request.mode == "live":
            snapshot = await bootstrap_live_snapshot(
                broker_settings=self._settings.broker,
                instrument_contracts=contracts,
            )
            session = create_live_session_impl(
                settings=self._settings,
                initial_snapshot=snapshot,
                instrument_contracts=contracts,
            )
            market_prices = {
                position.instrument_id: position.market_price for position in snapshot.positions
            }
        else:
            session = create_paper_session_impl(
                settings=self._settings,
                initial_cash=request.initial_cash,
                instrument_contracts=contracts,
            )
            market_prices = contract_reference_prices(contracts)

        await session.broker.connect()
        try:
            decisions = await run_passive_reprice_once(session=session, market_prices=market_prices)
            if decisions and session.lifecycle_feed is not None:
                events = await session.lifecycle_feed.drain_lifecycle_events()
                if events:
                    await session.coordinator.process_lifecycle_events(events)
        finally:
            await session.broker.disconnect()
        return {
            "passed": True,
            "mode": request.mode,
            "decisions": [_decision_payload(decision) for decision in decisions],
        }

    async def sweep_dead_letters(self, request: EventBusSweepRequest) -> tuple[int, int]:
        from quant_platform.bootstrap.broker import sweep_dead_letters

        return await sweep_dead_letters(self._settings, request.stream)


def _decision_payload(decision: PassiveRepriceDecision) -> dict[str, object]:
    return {
        "order_id": str(decision.order_id),
        "action": decision.action,
        "reason": decision.reason,
        "broker_order_id": decision.broker_order_id,
        "replacement_order_id": str(decision.replacement_order_id)
        if decision.replacement_order_id
        else None,
        "new_limit_price": str(decision.new_limit_price)
        if decision.new_limit_price is not None
        else None,
    }


__all__ = ["BrokerAdapters"]
