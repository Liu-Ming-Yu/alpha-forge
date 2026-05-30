"""Console bootstrap + effective-configuration routes.

Two read-only endpoints that exist purely to back the browser console:

* ``GET /console/info`` — **public** (no key).  The SPA shell loads before the
  operator has supplied a key, so it needs a tiny, secret-free bootstrap:
  whether auth is required, the API version, and the static mode/profile enums.

* ``GET /v1/config/effective`` — **protected**.  A whitelisted, secret-scrubbed
  snapshot of the running configuration for the Settings → "Modes & config"
  inspector.  Run mode is a launch-time concern (CLI/env), so this is an
  inspector, not a control surface.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext

_API_VERSION = "0.1.0"
RUN_MODES = ("shadow", "paper", "live")
EXECUTION_BACKENDS = ("simulated", "ib-paper")
PROFILES = ("paper", "live")

# Substrings that mark a config field as sensitive; matching keys are dropped
# from the effective-config snapshot regardless of nesting depth.
_SECRET_HINTS = (
    "key",
    "token",
    "secret",
    "password",
    "dsn",
    "url",
    "credential",
    "account",
)


def _is_secret(field_name: str) -> bool:
    lowered = field_name.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


# Embedded credentials in a URL/DSN value (``scheme://user:pass@host``) — a
# defence-in-depth backstop for secrets that slip past the key-name filter.
_CREDENTIAL_URL = re.compile(r"://[^/\s:@]+:[^/\s@]+@")


def _scrub(value: object) -> object:
    """Recursively drop secret-looking keys and redact credential-bearing
    string values from a JSON-able structure."""
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if not _is_secret(str(k))}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and _CREDENTIAL_URL.search(value):
        return "***redacted***"
    return value


def _section(settings: object, name: str) -> dict[str, object]:
    section = getattr(settings, name, None)
    if section is None or not hasattr(section, "model_dump"):
        return {}
    dumped = section.model_dump(mode="json")
    if not isinstance(dumped, dict):
        return {}
    scrubbed = _scrub(dumped)
    return scrubbed if isinstance(scrubbed, dict) else {}


def console_info_payload() -> dict[str, Any]:
    return {
        "api_version": _API_VERSION,
        "requires_auth": True,
        "product": "Quant Platform Operator Console",
        "run_modes": list(RUN_MODES),
        "execution_backends": list(EXECUTION_BACKENDS),
        "profiles": list(PROFILES),
    }


def effective_config_payload(ctx: OperatorApiRouteContext) -> dict[str, Any]:
    settings = ctx.settings
    storage = settings.storage
    broker = settings.broker
    production = getattr(settings, "production", None)
    deployment = {
        "paper_trading": bool(broker.paper_trading),
        "profile_preset": getattr(production, "profile_preset", None),
        "broker_host": broker.host,
        "broker_port": int(broker.port),
        "broker_client_id": int(broker.client_id),
        "primary_broker_path": getattr(broker, "primary_broker_path", None),
        "event_bus_backend": storage.event_bus_backend,
        "postgres_configured": bool(storage.postgres_dsn),
        "redis_configured": bool(storage.redis_url),
        "object_store_root": storage.object_store_root,
    }
    return {
        "as_of": ctx.clock.now().isoformat(),
        "deployment": deployment,
        "capabilities": ctx.capabilities_payload(),
        "alpha_source_weights": {
            str(source): float(weight)
            for source, weight in dict(settings.alpha.source_weights).items()
        },
        "sections": {
            name: _section(settings, name)
            for name in (
                "risk",
                "execution",
                "throttle",
                "cash",
                "liquidity",
                "vol_sizing",
                "regime",
                "production",
                "v2",
            )
        },
        "enums": {
            "run_modes": list(RUN_MODES),
            "execution_backends": list(EXECUTION_BACKENDS),
            "profiles": list(PROFILES),
        },
    }


def register_console_config_routes(app: FastAPI, ctx: OperatorApiRouteContext) -> None:
    @app.get("/console/info", include_in_schema=False)
    async def console_info() -> JSONResponse:
        return JSONResponse(content=console_info_payload())

    @app.get("/v1/config/effective", dependencies=ctx.protected_dependencies)
    async def effective_config() -> JSONResponse:
        return JSONResponse(content=effective_config_payload(ctx))
