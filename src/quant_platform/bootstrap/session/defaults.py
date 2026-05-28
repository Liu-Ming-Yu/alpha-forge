"""Session startup policy helpers."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.portfolio import RiskLimits
from quant_platform.services.signal_service.regime_detector import RegimeThresholds

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.application.runtime.state import Session
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)

_DEV_DEFAULT_PREFIXES: tuple[str, ...] = ("Simple", "InMemory")
_DEV_DEFAULT_ALLOWLIST: frozenset[str] = frozenset()


def session_default_fields(session: Session) -> dict[str, str]:
    """Map operator-facing plug-ins on ``session`` to runtime class names."""
    return {
        "event_bus": type(session.event_bus).__name__,
        "regime_detector": type(session.regime_detector).__name__
        if session.regime_detector is not None
        else "<none>",
        "order_repo": type(session.order_repo).__name__,
        "position_repo": type(session.position_repo).__name__,
        "performance_repo": type(session.performance_repo).__name__,
        "feature_repo": type(session.feature_repo).__name__,
        "text_event_store": type(session.text_event_store).__name__,
        "bar_store": type(session.bar_store).__name__,
        "audit_sink": type(session.audit_sink).__name__,
    }


def log_session_defaults(session: Session, *, mode: str) -> None:
    """Emit a structured ``session.defaults`` record listing the wiring."""
    log.info("session.defaults", mode=mode, **session_default_fields(session))


def assert_live_session_defaults(session: Session) -> None:
    """Fail closed when dev stubs leak into a live session."""
    if session.settings.allow_dev_defaults:
        log.warning(
            "session.allow_dev_defaults.enabled",
            detail="live session permitted to run with Simple*/InMemory* defaults",
        )
        return

    violations: dict[str, str] = {}
    for field_name, cls_name in session_default_fields(session).items():
        if cls_name in _DEV_DEFAULT_ALLOWLIST:
            continue
        if cls_name.startswith(_DEV_DEFAULT_PREFIXES):
            violations[field_name] = cls_name

    if violations:
        log.error("session.defaults.violations", **violations)
        raise RuntimeError(
            "Live session refuses to start with dev-stub defaults: "
            f"{violations}.  Wire a production implementation (Postgres / "
            f"Redis / MarketRegimeDetector) or, for a deliberate smoke-test, "
            f"set ``QP__ALLOW_DEV_DEFAULTS=true``."
        )


def regime_thresholds_from_settings(settings: PlatformSettings) -> RegimeThresholds:
    """Build a ``RegimeThresholds`` from ``settings.regime.thresholds``."""
    thresholds = settings.regime.thresholds
    return RegimeThresholds(
        crisis_vol=thresholds.crisis_vol,
        risk_off_vol=thresholds.risk_off_vol,
        low_vol=thresholds.low_vol,
        downtrend_z=thresholds.downtrend_z,
        uptrend_z=thresholds.uptrend_z,
        weak_breadth=thresholds.weak_breadth,
        strong_breadth=thresholds.strong_breadth,
    )


def risk_limits_from_settings(
    settings: PlatformSettings,
    strategy_run_id: uuid.UUID,
    effective_from: datetime,
) -> RiskLimits:
    risk_settings = settings.risk
    return RiskLimits(
        limits_id=uuid.uuid4(),
        strategy_run_id=strategy_run_id,
        effective_from=effective_from,
        max_single_name_weight=risk_settings.max_single_name_weight,
        max_sector_weight=risk_settings.max_sector_weight,
        max_gross_exposure=risk_settings.max_gross_exposure,
        max_daily_turnover=risk_settings.max_daily_turnover,
        min_cash_buffer=risk_settings.min_cash_buffer,
        max_drawdown_halt=risk_settings.max_drawdown_halt,
        vol_target_annualised=risk_settings.vol_target_annualised,
    )
