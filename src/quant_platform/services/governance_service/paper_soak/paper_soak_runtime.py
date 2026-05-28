"""Durable runtime evidence queries for paper-soak reports."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_platform.config import PlatformSettings


def build_kill_switch_store(_dsn: str) -> object:
    """Compatibility injection hook for tests; bootstrap supplies production stores."""
    raise RuntimeError("kill-switch store factory must be supplied by bootstrap")


def create_pg_engine(_dsn: str) -> object:
    """Compatibility injection hook for tests; bootstrap supplies production engines."""
    raise RuntimeError("Postgres engine factory must be supplied by bootstrap")


async def reconciliation_section(
    settings: PlatformSettings,
    *,
    kill_switch_store_factory: Callable[[str], object] | None = None,
) -> dict[str, Any]:
    """Build reconciliation drift evidence from durable kill-switch state."""
    if not settings.storage.postgres_dsn.strip():
        return {"drift_detected": False, "detail": "postgres not configured (in-memory mode)"}
    try:
        factory = kill_switch_store_factory or build_kill_switch_store
        store = cast("Any", factory(settings.storage.postgres_dsn))
        state = await store.get()
    except Exception as exc:
        return {"drift_detected": False, "detail": f"kill switch query failed: {exc}"}

    if state.active and (state.activated_by or "").lower() == "reconciliation":
        return {
            "drift_detected": True,
            "detail": (
                f"kill switch active, activated_by={state.activated_by!r}, reason={state.reason!r}"
            ),
            "kill_switch_active": state.active,
        }
    if state.active:
        detail = f"kill switch active (non-reconciliation), activated_by={state.activated_by!r}"
    else:
        detail = "kill switch clear"
    return {"drift_detected": False, "detail": detail, "kill_switch_active": state.active}


async def order_latency_section(
    settings: PlatformSettings,
    *,
    as_of: datetime,
    window_days: int,
    pg_engine_factory: Callable[[str], object] | None = None,
) -> dict[str, Any]:
    """Aggregate submit-to-fill latency from durable order state."""
    if not settings.storage.postgres_dsn.strip():
        return {
            "passed": False,
            "detail": "postgres not configured; durable order latency unavailable",
        }
    try:
        from sqlalchemy import text
    except ImportError:
        return {"passed": False, "detail": "sqlalchemy not available for latency query"}

    cutoff = (as_of - timedelta(days=max(1, window_days))).isoformat()
    try:
        factory = pg_engine_factory or create_pg_engine
        engine = cast("Any", factory(settings.storage.postgres_dsn))
        async with engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text(
                            """
                            SELECT
                              COUNT(*) AS sample_count,
                              AVG(
                                EXTRACT(EPOCH FROM (f.executed_at - o.created_at))
                              ) AS avg_fill_secs,
                              MAX(
                                EXTRACT(EPOCH FROM (f.executed_at - o.created_at))
                              ) AS max_fill_secs,
                              AVG(f.slippage_bps) AS avg_slippage_bps
                            FROM fill_events f
                            JOIN order_intents o ON o.order_id = f.order_id
                            WHERE f.executed_at >= :cutoff
                              AND o.created_at IS NOT NULL
                            """
                        ),
                        {"cutoff": cutoff},
                    )
                )
                .mappings()
                .first()
            )
    except Exception as exc:
        return {"passed": False, "detail": f"order latency query failed: {exc}"}

    if row is None or row.get("sample_count") in (None, 0):
        return {
            "passed": False,
            "sample_count": 0,
            "detail": "no fills observed in window",
            "window_days": window_days,
        }
    sample_count = int(row["sample_count"])
    avg_fill_secs = float(row["avg_fill_secs"]) if row["avg_fill_secs"] is not None else None
    max_fill_secs = float(row["max_fill_secs"]) if row["max_fill_secs"] is not None else None
    avg_slip = float(row["avg_slippage_bps"]) if row["avg_slippage_bps"] is not None else None
    passed = bool(sample_count > 0 and avg_fill_secs is not None)
    return {
        "passed": passed,
        "window_days": window_days,
        "sample_count": sample_count,
        "avg_fill_seconds": avg_fill_secs,
        "max_fill_seconds": max_fill_secs,
        "avg_slippage_bps": avg_slip,
    }


__all__ = ["order_latency_section", "reconciliation_section"]
