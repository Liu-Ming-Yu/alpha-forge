"""Pure paper-soak report section builders."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quant_platform.core.domain.production import (
        BrokerHealthObservation,
        DataHealthReport,
        NavSnapshot,
        PaperLifecycleObservation,
        SignalGateStatus,
    )


def broker_health_section(
    health: BrokerHealthObservation | None,
    *,
    as_of: datetime,
    stale_after: timedelta,
) -> dict[str, Any]:
    if health is None:
        return {"passed": False, "detail": "no broker health observation persisted"}
    fresh = health.observed_at >= as_of - stale_after
    passed = bool(health.status == "connected" and fresh)
    payload = asdict(health)
    payload["observed_at"] = health.observed_at.isoformat()
    if health.last_heartbeat_at is not None:
        payload["last_heartbeat_at"] = health.last_heartbeat_at.isoformat()
    payload["passed"] = passed
    payload["fresh"] = fresh
    return section_passed(payload)


def lifecycle_section(
    lifecycle: PaperLifecycleObservation | None,
    *,
    as_of: datetime,
    stale_after: timedelta,
) -> dict[str, Any]:
    if lifecycle is None:
        return {"passed": False, "detail": "no paper lifecycle observation persisted"}
    fresh = lifecycle.observed_at >= as_of - stale_after
    passed = bool(lifecycle.passed and fresh)
    payload = asdict(lifecycle)
    payload["observed_at"] = lifecycle.observed_at.isoformat()
    payload["instrument_id"] = str(lifecycle.instrument_id)
    payload["passed"] = passed
    payload["fresh"] = fresh
    return section_passed(payload)


def nav_section(snapshot: NavSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {"passed": False, "detail": "no nav snapshot persisted"}
    return {
        "snapshot_id": str(snapshot.snapshot_id),
        "strategy_run_id": str(snapshot.strategy_run_id),
        "as_of": snapshot.as_of.isoformat(),
        "net_asset_value": float(snapshot.net_asset_value),
        "gross_exposure": float(snapshot.gross_exposure),
        "cash": float(snapshot.cash),
        "source": snapshot.source,
        "passed": float(snapshot.net_asset_value) > 0,
    }


def data_health_section(report: DataHealthReport | None) -> dict[str, Any]:
    if report is None:
        return {"passed": False, "detail": "data-health report unavailable"}
    return {
        "passed": report.passed,
        "instruments_checked": report.instruments_checked,
        "instruments_with_bars": report.instruments_with_bars,
        "instruments_with_liquidity": report.instruments_with_liquidity,
        "stale_instruments": report.stale_instruments,
        "coverage_pct": report.coverage_pct,
        "liquidity_coverage_pct": report.liquidity_coverage_pct,
    }


def signal_gate_section(status: SignalGateStatus | None) -> dict[str, Any]:
    if status is None:
        return {"passed": False, "detail": "no signal gate evidence configured"}
    return {
        "passed": status.passed,
        "signal_name": status.signal_name,
        "signal_type": status.signal_type,
        "as_of": status.as_of.isoformat(),
        "rolling_ic": status.rolling_ic,
        "observations": status.observations,
        "negative_streak": status.negative_streak,
        "max_drawdown": status.max_drawdown,
        "max_turnover": status.max_turnover,
    }


def section_passed(detail: dict[str, Any]) -> dict[str, Any]:
    """Return the dict with a normalised ``passed`` boolean."""
    out = {k: _decimal(v) for k, v in detail.items()}
    out["passed"] = bool(detail.get("passed", False))
    return out


def midnight_utc(value: datetime) -> datetime:
    """Round a UTC timestamp down to the start of the day."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    value_utc = value.astimezone(UTC)
    today = date(value_utc.year, value_utc.month, value_utc.day)
    return datetime.combine(today, datetime.min.time(), tzinfo=UTC)


def _decimal(value: object) -> object:
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
