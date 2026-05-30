"""Operator API settings."""

from __future__ import annotations

from pydantic import BaseModel


class ApiSettings(BaseModel):
    """Operator API authentication parameters.

    ``operator_api_key`` gates every protected endpoint; public ``/health`` is
    limited to liveness and detailed broker state lives under authenticated
    ``/health/details``.  Starting the API without a key requires *two*
    explicit flags:

    1. ``allow_unauthenticated=true`` opts out of the default-secure gate.
    2. ``acknowledge_unauthenticated_risk=true`` acknowledges the operator
       understands this leaves every protected endpoint open.

    Requiring two independently-named flags (R-OBS-02) means a single
    misconfigured env var cannot expose the full protected surface; the
    operator must flip both knobs deliberately.

    ``rate_limit_per_minute`` applies per API key (or per client host when
    the escape hatch is in effect).
    """

    operator_api_key: str = ""
    allow_unauthenticated: bool = False
    acknowledge_unauthenticated_risk: bool = False
    rate_limit_per_minute: int = 120
    cors_allow_origins: str = ""
    expose_metrics: bool = False
    console_dist_dir: str = ""
    """Filesystem path to the built operator console SPA (``ui/dist``).

    Empty (default) resolves to ``<repo_root>/ui/dist``.  Set
    ``QP__API__CONSOLE_DIST_DIR`` to serve a console built elsewhere.  When the
    directory is absent the API serves a friendly build-instructions page at
    ``/app`` instead of failing — the JSON API is unaffected.
    """
    enable_command_execution: bool = False
    """Allow the console to RUN CLI commands as subprocess jobs (default off).

    The command *catalog* is always browsable, but ``POST /v1/commands/run`` is
    refused unless this is true.  This is opt-in because it lets an
    authenticated operator launch any ``python -m quant_platform`` command —
    including live trading and migrations — from the browser.  Set
    ``QP__API__ENABLE_COMMAND_EXECUTION=true`` to enable; dangerous commands
    still require a typed confirmation.
    """

    def assert_api_ready(self) -> None:
        """Raise ValueError if the API cannot start safely.

        Call from the FastAPI lifespan or app factory, not at config load time,
        so that services that don't start the API can still construct settings.
        """
        if not self.operator_api_key and not self.allow_unauthenticated:
            raise ValueError(
                "operator_api_key must be set, or allow_unauthenticated=true must be explicitly "
                "enabled (along with acknowledge_unauthenticated_risk=true). "
                "Set QP__API__OPERATOR_API_KEY or QP__API__ALLOW_UNAUTHENTICATED=true."
            )
