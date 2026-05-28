"""Shared route context for the operator API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from starlette.responses import Response

_REPO_TIMEOUT_SECONDS = 10.0


@dataclass(slots=True)
class OperatorApiRouteContext:
    """Dependencies and helpers shared by route groups.

    Keeping this as an explicit context makes the FastAPI handlers thin without
    asking them to reach back into bootstrap modules or concrete repositories.
    """

    settings: Any
    clock: Any
    ledger: Any
    throttle_policy: Any
    selected_audit_sink: Any
    v2_auth_repo: Any
    builder: Any
    kill_switch_store: Any
    operator_api_key: str
    research_queries: Any
    shutting_down: list[bool]
    protected_dependencies: list[Any]
    repo_timeout_seconds: float = _REPO_TIMEOUT_SECONDS

    def json_payload(self, response: Response) -> dict[str, Any]:
        body = response.body
        if isinstance(body, memoryview):
            body = body.tobytes()
        payload = json.loads(body.decode("utf-8")) if isinstance(body, bytes) else json.loads(body)
        if not isinstance(payload, dict):
            raise TypeError("operator API response payload must be a JSON object")
        return payload

    def capabilities_payload(self) -> dict[str, Any]:
        write_controls = {
            "kill_switch_clear": self.kill_switch_store is not None,
            "trading": False,
            "model_promotion": False,
            "alpha_promotion": False,
        }
        return {
            "api_version": "0.1.0",
            "auth": {
                "mode": "api_key" if self.operator_api_key else "unauthenticated",
                "v2_operator_auth": self.v2_auth_repo is not None,
                "roles_advertised": (["viewer", "operator", "admin"] if self.v2_auth_repo else []),
            },
            "features": {
                "dashboard_summary": True,
                "strategy_run_discovery": bool(self.settings.storage.postgres_dsn),
                "research_campaigns": True,
                "feature_audits": True,
                "prometheus_metrics": bool(getattr(self.settings.api, "expose_metrics", False)),
                "v2_enabled": bool(self.settings.v2.enabled),
                "v2_postgres": self.v2_auth_repo is not None,
                "event_bus_backend": self.settings.storage.event_bus_backend,
                "postgres": bool(self.settings.storage.postgres_dsn),
                "redis": bool(self.settings.storage.redis_url),
                "production_candidate": True,
                "readiness_snapshot": self.v2_auth_repo is not None,
            },
            "write_controls": write_controls,
            "unsupported_features": [
                name for name, enabled in write_controls.items() if not enabled
            ],
        }

    async def list_research_campaigns(self, limit: int = 20) -> dict[str, Any]:
        return dict(await self.research_queries.list_research_campaigns(limit=limit))

    async def read_research_campaign(self, run_id: str) -> dict[str, Any] | None:
        payload = await self.research_queries.read_research_campaign(run_id)
        return dict(payload) if payload is not None else None

    async def list_feature_audits(
        self,
        *,
        feature_name: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        return dict(
            await self.research_queries.list_feature_audits(
                feature_name=feature_name,
                limit=limit,
            )
        )
