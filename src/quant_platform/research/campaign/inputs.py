"""Research-campaign input calibration and parsing helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from quant_platform.application.errors import OperatorUsageError

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)


async def _observed_slippage_bps(settings: PlatformSettings) -> float | None:
    """Aggregate average paper-fill slippage across the most recent runs."""
    if not settings.storage.postgres_dsn.strip():
        return None
    try:
        from sqlalchemy import text

        from quant_platform.infrastructure.postgres.repositories import create_pg_engine
    except ImportError:
        return None
    try:
        engine = create_pg_engine(settings.storage.postgres_dsn)
        async with engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text(
                            """
                            SELECT AVG(slippage_bps) AS avg_slippage
                            FROM (
                                SELECT slippage_bps
                                FROM fill_events
                                WHERE slippage_bps IS NOT NULL
                                ORDER BY executed_at DESC
                                LIMIT 5000
                            ) recent_fills
                            """
                        )
                    )
                )
                .mappings()
                .first()
            )
    except Exception as exc:
        log.warning("research_campaign.slippage_query_failed", error=str(exc))
        return None
    if row is None or row["avg_slippage"] is None:
        return None
    try:
        return float(row["avg_slippage"])
    except (TypeError, ValueError):
        return None


def _parse_paper_source_weights(
    settings: PlatformSettings,
    raw: str,
) -> dict[str, float]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise OperatorUsageError("--paper-source-weights-json must be a JSON object")
    weights = {str(key): float(value) for key, value in payload.items()}
    if any(value < 0 for value in weights.values()):
        raise OperatorUsageError("paper source weights must be non-negative")
    if not weights or sum(weights.values()) <= 0:
        raise OperatorUsageError("paper source weights must contain a positive total weight")
    non_classical = sum(
        value for source, value in weights.items() if source not in {"classical", "primary"}
    )
    cap = float(settings.alpha.paper_max_non_classical_weight)
    if non_classical > cap:
        raise OperatorUsageError(
            "paper non-classical source weight exceeds "
            f"QP__ALPHA__PAPER_MAX_NON_CLASSICAL_WEIGHT={cap:.2f}"
        )
    return weights
