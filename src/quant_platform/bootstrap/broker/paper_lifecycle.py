"""IB paper submit/open/cancel lifecycle probe."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.bootstrap.broker.probe import (
    broker_gate_settings,
    classify_broker_probe_failure,
)
from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema
from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.production import PaperLifecycleObservation
from quant_platform.infrastructure.performance import build_performance_repository

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.config import PlatformSettings


async def ib_paper_lifecycle(
    settings: PlatformSettings,
    *,
    contracts: Mapping[uuid.UUID, dict[str, object]],
    instrument_id: uuid.UUID,
    max_notional_usd: Decimal,
    max_allowed_notional: Decimal,
) -> dict[str, object]:
    """Run a paper-only non-marketable submit/open/cancel lifecycle probe."""

    if not settings.broker.paper_trading:
        raise ValueError("ib-paper-lifecycle refuses to run unless paper_trading=true")
    if max_notional_usd <= 0 or max_notional_usd > max_allowed_notional:
        raise ValueError(f"--max-notional-usd must be > 0 and <= {max_allowed_notional}")

    await verify_postgres_schema(settings)
    contract = contracts.get(instrument_id)
    if contract is None:
        raise ValueError(f"instrument_id {instrument_id} is not present in contracts")
    con_id = contract.get("con_id")
    if not isinstance(con_id, int) or con_id <= 0:
        raise ValueError(f"instrument_id {instrument_id} requires a positive integer con_id")

    broker_settings = broker_gate_settings(settings)
    if not broker_settings.host or broker_settings.port <= 0:
        raise ValueError("broker host and port must be configured")

    limit_price = paper_lifecycle_limit_price(contract, max_notional_usd)
    if limit_price <= 0:
        raise ValueError("computed paper lifecycle limit price must be positive")
    quantity = max(1, int(max_notional_usd / limit_price))
    while Decimal(quantity) * limit_price > max_notional_usd and quantity > 1:
        quantity -= 1

    order_id = uuid.uuid4()
    intent = OrderIntent(
        order_id=order_id,
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=datetime.now(tz=UTC),
        limit_price=limit_price,
    )

    from quant_platform.services.execution_service.gateways import broker_gateway

    gateway = broker_gateway.IBGatewayBrokerGateway(
        settings=broker_settings,
        instrument_contracts=dict(contracts),
    )
    observed_at = datetime.now(tz=UTC)
    broker_order_id = ""
    ack_status = "skipped"
    cancel_status = "skipped"
    stale_open_order_count = 0
    status = "failed"
    detail = ""
    try:
        await gateway.connect()
        ack = await gateway.place_order(intent)
        broker_order_id = ack.broker_order_id
        ack_status = "ok"
        open_orders = await gateway.fetch_open_orders()
        if not any(
            order.order_id == order_id or order.broker_order_id == broker_order_id
            for order in open_orders
        ):
            raise RuntimeError("submitted paper lifecycle order was not returned by open orders")
        await gateway.cancel_order(broker_order_id)
        cancel_status = "ok"
        remaining = await gateway.fetch_open_orders()
        stale_open_order_count = sum(
            1
            for order in remaining
            if order.order_id == order_id or order.broker_order_id == broker_order_id
        )
        if stale_open_order_count:
            raise RuntimeError(f"cancelled order still present in open orders: {broker_order_id}")
        status = "passed"
        detail = "paper lifecycle submit/open/cancel/reconcile passed"
    except Exception as exc:
        detail = f"{classify_broker_probe_failure(exc)}: {exc}"
    finally:
        try:
            await gateway.disconnect()
        except Exception as exc:
            detail = f"{detail}; disconnect_error={exc}" if detail else f"disconnect_error={exc}"

    observation = PaperLifecycleObservation(
        observed_at=observed_at,
        status=status,
        host=broker_settings.host,
        port=broker_settings.port,
        client_id=broker_settings.client_id,
        instrument_id=instrument_id,
        broker_order_id=broker_order_id,
        max_notional_usd=max_notional_usd,
        limit_price=limit_price,
        quantity=quantity,
        ack_status=ack_status,
        cancel_status=cancel_status,
        stale_open_order_count=stale_open_order_count,
        detail=detail,
    )
    if settings.storage.postgres_dsn:
        repo = build_performance_repository(settings.storage.postgres_dsn)
        await repo.save_paper_lifecycle(observation)

    return {
        "passed": observation.passed,
        "status": status,
        "host": broker_settings.host,
        "port": broker_settings.port,
        "client_id": broker_settings.client_id,
        "instrument_id": instrument_id,
        "broker_order_id": broker_order_id,
        "max_notional_usd": max_notional_usd,
        "limit_price": limit_price,
        "quantity": quantity,
        "ack_status": ack_status,
        "cancel_status": cancel_status,
        "stale_open_order_count": stale_open_order_count,
        "paper_trading": broker_settings.paper_trading,
        "detail": detail,
        "observed_at": observed_at,
    }


def paper_lifecycle_limit_price(
    contract: Mapping[str, object],
    max_notional_usd: Decimal,
) -> Decimal:
    raw_last_close = contract.get("last_close")
    if raw_last_close is None:
        raise ValueError("paper lifecycle contract requires last_close for non-marketable limit")
    last_close = Decimal(str(raw_last_close))
    if last_close <= 0:
        raise ValueError("paper lifecycle contract last_close must be positive")
    return min(last_close * Decimal("0.50"), max_notional_usd).quantize(Decimal("0.01"))


__all__ = ["ib_paper_lifecycle", "paper_lifecycle_limit_price"]
