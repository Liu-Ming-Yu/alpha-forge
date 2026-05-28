"""Startup security checks for the operator API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings

_SECURITY_LOG = structlog.get_logger("quant_platform.operator_api")


def resolve_operator_api_key(settings: PlatformSettings) -> str:
    """Validate operator API authentication settings and return the active key."""
    operator_api_key = settings.api.operator_api_key.strip()
    if operator_api_key:
        return operator_api_key

    if not settings.api.allow_unauthenticated:
        raise RuntimeError(
            "Operator API refuses to start without QP__API__OPERATOR_API_KEY. "
            "Set a key, or set QP__API__ALLOW_UNAUTHENTICATED=true AND "
            "QP__API__ACKNOWLEDGE_UNAUTHENTICATED_RISK=true to explicitly "
            "opt out (NOT recommended outside local development)."
        )
    if not settings.api.acknowledge_unauthenticated_risk:
        raise RuntimeError(
            "QP__API__ALLOW_UNAUTHENTICATED=true also requires "
            "QP__API__ACKNOWLEDGE_UNAUTHENTICATED_RISK=true.  This dual "
            "opt-in exists so a single misconfigured flag cannot expose "
            "the protected operator API.  See docs/runbooks/startup-and-"
            "migrations.md for the rationale."
        )
    _SECURITY_LOG.warning(
        "operator_api.SECURITY_WARNING.unauthenticated_enabled",
        detail=(
            "Starting without an API key because QP__API__ALLOW_UNAUTHENTICATED=true"
            " and QP__API__ACKNOWLEDGE_UNAUTHENTICATED_RISK=true."
            " Protected endpoints will be served with no authentication."
        ),
    )
    return operator_api_key
