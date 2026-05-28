"""Compatibility exports for order domain models.

Order value objects are split into lifecycle enums, broker/fill records,
intent/OMS lifecycle state, and execution-quality DTOs.  This module remains
the stable import surface for services, adapters, and tests.
"""

from __future__ import annotations

from quant_platform.core.domain.orders.broker import BrokerOrder, FillEvent
from quant_platform.core.domain.orders.enums import (
    ExecutionTactic,
    OrderSide,
    OrderStateEventType,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.orders.execution import ExecutionQualityReport, VenueRoute
from quant_platform.core.domain.orders.intent import (
    CancelReplaceRequest,
    OrderIntent,
    OrderStateEvent,
)

__all__ = [
    "BrokerOrder",
    "CancelReplaceRequest",
    "ExecutionQualityReport",
    "ExecutionTactic",
    "FillEvent",
    "OrderIntent",
    "OrderSide",
    "OrderStateEvent",
    "OrderStateEventType",
    "OrderStatus",
    "OrderType",
    "TimeInForce",
    "VenueRoute",
]
