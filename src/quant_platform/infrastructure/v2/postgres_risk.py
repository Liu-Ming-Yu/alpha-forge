"""Postgres-backed V2 portfolio risk repositories."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.v2.portfolio_json import (
    covariance_to_json as _covariance_to_json,
)
from quant_platform.infrastructure.v2.portfolio_json import (
    factor_exposures_to_json as _factor_exposures_to_json,
)
from quant_platform.infrastructure.v2.portfolio_json import (
    scenarios_to_json as _scenarios_to_json,
)
from quant_platform.infrastructure.v2.postgres_mappers import _row_to_risk_model

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.domain.portfolio import PortfolioRiskModel, RiskSnapshot


class PostgresPortfolioRiskModelRepository:
    """Postgres-backed risk model and risk snapshot repository."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save_risk_model(self, model: PortfolioRiskModel) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO portfolio_risk_models
                        (model_id, as_of, dataset_id, schema_hash, covariance_json,
                         factor_exposures_json, scenarios_json)
                    VALUES
                        (:model_id, :as_of, :dataset_id, :schema_hash,
                         CAST(:covariance_json AS JSONB),
                         CAST(:factor_exposures_json AS JSONB),
                         CAST(:scenarios_json AS JSONB))
                    ON CONFLICT (model_id) DO NOTHING
                """),
                {
                    "model_id": model.model_id,
                    "as_of": model.as_of,
                    "dataset_id": model.dataset_id,
                    "schema_hash": model.schema_hash,
                    "covariance_json": json.dumps(_covariance_to_json(model.covariance)),
                    "factor_exposures_json": json.dumps(
                        _factor_exposures_to_json(model.factor_exposures)
                    ),
                    "scenarios_json": json.dumps(_scenarios_to_json(model.scenarios)),
                },
            )

    async def latest_risk_model(self, *, as_of: datetime) -> PortfolioRiskModel | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT *
                            FROM portfolio_risk_models
                            WHERE as_of <= :as_of
                            ORDER BY as_of DESC
                            LIMIT 1
                        """),
                        {"as_of": as_of},
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_risk_model(row) if row else None

    async def save_risk_snapshot(self, snapshot: RiskSnapshot) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO portfolio_risk_snapshots
                        (snapshot_id, strategy_run_id, as_of, factor_exposures_json,
                         stress_results_json, cvar, gross_exposure, net_exposure, passed)
                    VALUES
                        (:snapshot_id, :strategy_run_id, :as_of,
                         CAST(:factor_exposures_json AS JSONB),
                         CAST(:stress_results_json AS JSONB), :cvar,
                         :gross_exposure, :net_exposure, :passed)
                    ON CONFLICT (snapshot_id) DO NOTHING
                """),
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "strategy_run_id": snapshot.strategy_run_id,
                    "as_of": snapshot.as_of,
                    "factor_exposures_json": json.dumps(snapshot.factor_exposures, default=str),
                    "stress_results_json": json.dumps(snapshot.stress_results, default=str),
                    "cvar": snapshot.cvar,
                    "gross_exposure": snapshot.gross_exposure,
                    "net_exposure": snapshot.net_exposure,
                    "passed": snapshot.passed,
                },
            )
