"""IB order mapping helpers."""

from __future__ import annotations

from ibapi.order import Order as IBOrder

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)


def build_ib_order(order: OrderIntent) -> IBOrder:
    """Translate a domain OrderIntent into an ibapi Order."""
    ib_order = IBOrder()
    ib_order.action = "BUY" if order.side == OrderSide.BUY else "SELL"
    ib_order.totalQuantity = order.quantity
    ib_order.orderRef = str(order.order_id)
    ib_order.transmit = True

    # ibapi 9.81 defaults these SMART-routing flags to True, but
    # current IB Gateway builds reject them for stock orders (error 10268).
    if hasattr(ib_order, "eTradeOnly"):
        ib_order.eTradeOnly = False
    if hasattr(ib_order, "firmQuoteOnly"):
        ib_order.firmQuoteOnly = False

    if order.order_type == OrderType.LIMIT:
        ib_order.orderType = "LMT"
        ib_order.lmtPrice = float(order.limit_price) if order.limit_price else 0.0
    elif order.order_type == OrderType.MOC:
        ib_order.orderType = "MOC"
    elif order.order_type == OrderType.LOC:
        ib_order.orderType = "LOC"
        ib_order.lmtPrice = float(order.limit_price) if order.limit_price else 0.0
    else:
        ib_order.orderType = "MKT"

    tif_map = {
        TimeInForce.DAY: "DAY",
        TimeInForce.GTC: "GTC",
        TimeInForce.IOC: "IOC",
        TimeInForce.FOK: "FOK",
    }
    ib_order.tif = tif_map.get(order.time_in_force, "DAY")
    return ib_order
