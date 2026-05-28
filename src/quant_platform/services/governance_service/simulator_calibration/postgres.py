"""Postgres loader for simulator calibration observations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from quant_platform.services.governance_service.simulator_calibration.models import (
    FillObservation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_platform.config import PlatformSettings


def create_pg_engine(_dsn: str) -> object:
    """Compatibility injection hook for tests; bootstrap supplies production engines."""
    raise RuntimeError("Postgres engine factory must be supplied by bootstrap")


async def load_paper_fills_from_postgres(
    settings: PlatformSettings,
    *,
    as_of: datetime,
    lookback_days: int,
    pg_engine_factory: Callable[[str], object] | None = None,
) -> list[FillObservation]:
    """Load recent paper fills as :class:`FillObservation` rows."""
    if not settings.storage.postgres_dsn.strip():
        return []
    try:
        from sqlalchemy import text
    except ImportError:
        return []
    cutoff = as_of.astimezone(UTC).isoformat()
    factory = pg_engine_factory or create_pg_engine
    engine = cast("Any", factory(settings.storage.postgres_dsn))
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        """
                        SELECT
                          coalesce(eqr.tactic, 'unknown')                AS tactic,
                          o.side                                         AS side,
                          o.order_type                                   AS order_type,
                          o.quantity                                     AS quantity,
                          coalesce(i.adv_shares_20d, 0)::float           AS adv_shares_20d,
                          coalesce(eqr.participation_rate, 0)::float     AS participation_rate,
                          coalesce(f.slippage_bps, 0)::float             AS slippage_bps,
                          f.executed_at                                  AS executed_at
                        FROM fill_events f
                        JOIN order_intents o ON o.order_id = f.order_id
                        LEFT JOIN execution_quality_reports eqr ON eqr.order_id = f.order_id
                        LEFT JOIN instruments i ON i.instrument_id = o.instrument_id
                        WHERE f.executed_at >= cast(:cutoff AS timestamptz)
                              - make_interval(days => :lookback_days)
                          AND f.slippage_bps IS NOT NULL
                        ORDER BY f.executed_at DESC
                        LIMIT 50000
                        """
                    ),
                    {"cutoff": cutoff, "lookback_days": max(1, lookback_days)},
                )
            )
            .mappings()
            .all()
        )
    out: list[FillObservation] = []
    for row in rows:
        out.append(
            FillObservation(
                tactic=str(row["tactic"]),
                side=str(row["side"]),
                quantity=int(row["quantity"]),
                adv_shares_20d=float(row["adv_shares_20d"]),
                spread_bps=0.0,
                slippage_bps=float(row["slippage_bps"]),
                executed_at=row["executed_at"],
                order_type=str(row["order_type"]),
            )
        )
    return out


__all__ = ["load_paper_fills_from_postgres"]
