"""Runtime hydration and preflight helpers for sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import structlog

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from datetime import datetime

    from quant_platform.application.runtime.state import Session

log = structlog.get_logger(__name__)


class KillSwitchState(Protocol):
    active: bool
    reason: str
    activated_at: datetime


class KillSwitchStateStore(Protocol):
    def get(self) -> Awaitable[KillSwitchState | None]: ...


async def hydrate_session_state(session: Session) -> None:
    """Restore durable runtime state from backing stores."""
    if session._state_hydrated:
        return
    durable_runtime_state = bool(session.settings.storage.postgres_dsn)
    store = session.kill_switch_store
    if store is not None:
        try:
            state = await cast("KillSwitchStateStore", store).get()
        except Exception as exc:  # pragma: no cover - connectivity
            log.error("session.kill_switch.hydrate_failed", error=str(exc))
            if durable_runtime_state:
                session._state_hydrated = False
                raise RuntimeError("failed to hydrate durable kill-switch state") from exc
            state = None
        if state is not None and hasattr(session.execution_policy, "hydrate_kill_switch"):
            session.execution_policy.hydrate_kill_switch(
                active=bool(state.active),
                reason=state.reason,
            )
            if state.active:
                log.warning(
                    "session.kill_switch.hydrated_active",
                    reason=state.reason,
                    activated_at=str(state.activated_at),
                )

    coordinator = getattr(session, "coordinator", None)
    if coordinator is not None and hasattr(coordinator, "hydrate"):
        try:
            await coordinator.hydrate()
        except Exception as exc:  # pragma: no cover - connectivity
            log.error("session.coordinator.hydrate_failed", error=str(exc))
            if durable_runtime_state:
                session._state_hydrated = False
                raise RuntimeError("failed to hydrate durable coordinator state") from exc
    session._state_hydrated = True


async def model_registry_preflight(
    session: Session,
    *,
    strategy_name: str,
    engine_version: str,
    require_match: bool | None = None,
) -> None:
    """Compare the active RegisteredModel against the running engine."""
    strict = (
        session.settings.risk.require_registered_model_match
        if require_match is None
        else require_match
    )
    if not session.settings.storage.postgres_dsn:
        if require_match is True:
            raise RuntimeError(
                "Model registry preflight: Postgres DSN is required for "
                f"strategy={strategy_name!r} when registered model matching is enforced."
            )
        log.info(
            "session.model.preflight.skipped",
            reason="no postgres registry configured",
        )
        return

    from quant_platform.infrastructure.postgres.model_registry import (
        build_model_registry,
    )

    registry = build_model_registry(session.settings.storage.postgres_dsn)
    try:
        active = await registry.get_active_model(strategy_name)
    except Exception as exc:  # pragma: no cover - connectivity
        log.error("session.model.preflight.lookup_failed", error=str(exc))
        if strict:
            raise
        return

    if active is None:
        if strict:
            raise RuntimeError(
                "Model registry preflight: no active RegisteredModel for "
                f"strategy={strategy_name!r}.  Promote a model or set "
                "QP__RISK__REQUIRE_REGISTERED_MODEL_MATCH=false to bypass."
            )
        log.warning(
            "session.model.preflight.no_active_model",
            strategy_name=strategy_name,
        )
        return

    if active.model_version != engine_version:
        if strict:
            raise RuntimeError(
                "Model registry preflight mismatch: active model_version="
                f"{active.model_version!r} != engine_version={engine_version!r}"
                f" for strategy={strategy_name!r}."
            )
        log.warning(
            "session.model.preflight.mismatch",
            strategy_name=strategy_name,
            model_version=active.model_version,
            engine_version=engine_version,
        )
        return

    log.info(
        "session.model.preflight.ok",
        strategy_name=strategy_name,
        model_version=active.model_version,
    )
