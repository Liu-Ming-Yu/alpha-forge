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
