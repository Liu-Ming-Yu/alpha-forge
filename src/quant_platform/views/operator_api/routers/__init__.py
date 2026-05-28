"""Operator API route groups.

The FastAPI application factory owns startup, auth, middleware, and dependency
composition.  These modules own read-only route registration for bounded
operator surfaces.
"""

from __future__ import annotations

from quant_platform.views.operator_api.routers.cash_orders_audit import (
    register_cash_orders_audit_routes,
)
from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext
from quant_platform.views.operator_api.routers.dashboard import register_dashboard_routes
from quant_platform.views.operator_api.routers.health import register_health_routes
from quant_platform.views.operator_api.routers.research_evidence import (
    register_research_evidence_routes,
)
from quant_platform.views.operator_api.routers.strategy_state import (
    register_strategy_state_routes,
)

__all__ = [
    "OperatorApiRouteContext",
    "register_cash_orders_audit_routes",
    "register_dashboard_routes",
    "register_health_routes",
    "register_research_evidence_routes",
    "register_strategy_state_routes",
]
