"""Persistence for ensemble signal contribution attribution."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from quant_platform.core.domain.production import SignalContribution
from quant_platform.infrastructure.postgres.row_coercion import require_datetime, require_float

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.engine import RowMapping
    from sqlalchemy.ext.asyncio import AsyncEngine


class InMemorySignalContributionRepository:
    """In-memory contribution repository for tests and local runs."""

    def __init__(self) -> None:
        self._rows: dict[uuid.UUID, SignalContribution] = {}

    async def save_signal_contributions(
        self,
        contributions: list[SignalContribution],
    ) -> None:
        for contribution in contributions:
            self._rows.setdefault(contribution.contribution_id, contribution)

    async def list_signal_contributions(
        self,
        *,
        strategy_run_id: uuid.UUID | None = None,
        score_id: uuid.UUID | None = None,
        instrument_id: uuid.UUID | None = None,
        limit: int = 500,
    ) -> list[SignalContribution]:
        rows = list(self._rows.values())
        if strategy_run_id is not None:
            rows = [row for row in rows if row.strategy_run_id == strategy_run_id]
        if score_id is not None:
            rows = [row for row in rows if row.score_id == score_id]
        if instrument_id is not None:
            rows = [row for row in rows if row.instrument_id == instrument_id]
        rows.sort(key=lambda row: row.as_of, reverse=True)
        return rows[: max(1, limit)]


class PostgresSignalContributionRepository:
    """PostgreSQL-backed contribution repository."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save_signal_contributions(
        self,
        contributions: list[SignalContribution],
    ) -> None:
        if not contributions:
            return
        async with self._engine.begin() as conn:
            for item in contributions:
                await conn.execute(
                    text("""
                        INSERT INTO signal_contributions
                            (contribution_id, score_id, strategy_run_id, instrument_id,
                             as_of, source, source_model_version, raw_score,
                             normalized_score, blend_weight, confidence,
                             feature_vector_id, promotion_state)
                        VALUES
                            (:contribution_id, :score_id, :strategy_run_id, :instrument_id,
                             :as_of, :source, :source_model_version, :raw_score,
                             :normalized_score, :blend_weight, :confidence,
                             :feature_vector_id, :promotion_state)
                        ON CONFLICT (contribution_id) DO NOTHING
                    """),
                    _params(item),
                )

    async def list_signal_contributions(
        self,
        *,
        strategy_run_id: uuid.UUID | None = None,
        score_id: uuid.UUID | None = None,
        instrument_id: uuid.UUID | None = None,
        limit: int = 500,
    ) -> list[SignalContribution]:
        params: dict[str, object] = {
            "filter_strategy_run": strategy_run_id is not None,
            "strategy_run_id": str(strategy_run_id) if strategy_run_id is not None else None,
            "filter_score": score_id is not None,
            "score_id": str(score_id) if score_id is not None else None,
            "filter_instrument": instrument_id is not None,
            "instrument_id": str(instrument_id) if instrument_id is not None else None,
            "limit": max(1, limit),
        }
        query = text("""
                            SELECT contribution_id, score_id, strategy_run_id, instrument_id,
                                   as_of, source, source_model_version, raw_score,
                                   normalized_score, blend_weight, confidence,
                                   feature_vector_id, promotion_state
                            FROM signal_contributions
                            WHERE (
                                :filter_strategy_run IS FALSE
                                OR strategy_run_id = CAST(:strategy_run_id AS UUID)
                            )
                              AND (
                                :filter_score IS FALSE
                                OR score_id = CAST(:score_id AS UUID)
                              )
                              AND (
                                :filter_instrument IS FALSE
                                OR instrument_id = CAST(:instrument_id AS UUID)
                              )
                            ORDER BY as_of DESC
                            LIMIT :limit
                            """)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query, params)).mappings().all()
        return [_row_to_contribution(row) for row in rows]


def build_signal_contribution_repository(
    dsn: str,
    engine: AsyncEngine | None = None,
) -> InMemorySignalContributionRepository | PostgresSignalContributionRepository:
    if not dsn:
        return InMemorySignalContributionRepository()
    if engine is None:
        from quant_platform.infrastructure.postgres.repositories import create_pg_engine

        engine = create_pg_engine(dsn)
    return PostgresSignalContributionRepository(engine)


def _params(item: SignalContribution) -> dict[str, object]:
    return {
        "contribution_id": item.contribution_id,
        "score_id": item.score_id,
        "strategy_run_id": item.strategy_run_id,
        "instrument_id": item.instrument_id,
        "as_of": item.as_of,
        "source": item.source,
        "source_model_version": item.source_model_version,
        "raw_score": item.raw_score,
        "normalized_score": item.normalized_score,
        "blend_weight": item.blend_weight,
        "confidence": item.confidence,
        "feature_vector_id": item.feature_vector_id,
        "promotion_state": item.promotion_state,
    }


def _row_to_contribution(row: Mapping[str, Any] | RowMapping) -> SignalContribution:
    data = row
    return SignalContribution(
        contribution_id=uuid.UUID(str(data["contribution_id"])),
        score_id=uuid.UUID(str(data["score_id"])),
        strategy_run_id=uuid.UUID(str(data["strategy_run_id"])),
        instrument_id=uuid.UUID(str(data["instrument_id"])),
        as_of=require_datetime(data, "as_of"),
        source=str(data["source"]),
        source_model_version=str(data["source_model_version"]),
        raw_score=require_float(data["raw_score"], name="raw_score"),
        normalized_score=require_float(data["normalized_score"], name="normalized_score"),
        blend_weight=require_float(data["blend_weight"], name="blend_weight"),
        confidence=require_float(data["confidence"], name="confidence"),
        feature_vector_id=uuid.UUID(str(data["feature_vector_id"]))
        if data.get("feature_vector_id")
        else None,
        promotion_state=str(data["promotion_state"]),
    )
