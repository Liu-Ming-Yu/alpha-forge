"""Repositories for multi-engine governance state."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.core.domain.production import (
    CombinedPortfolioTarget,
    EngineBudget,
    EngineTargetContribution,
    OrderAllocation,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class InMemoryMultiEngineGovernanceRepository:
    """In-memory governance repository for tests and local shadow runs."""

    def __init__(self) -> None:
        self._budgets: dict[str, EngineBudget] = {}
        self._targets: dict[uuid.UUID, CombinedPortfolioTarget] = {}
        self._allocations: dict[uuid.UUID, list[OrderAllocation]] = {}

    async def save_engine_budget(self, budget: EngineBudget) -> None:
        self._budgets[budget.engine_name] = budget

    async def list_engine_budgets(self) -> list[EngineBudget]:
        return sorted(self._budgets.values(), key=lambda row: row.engine_name)

    async def save_combined_target(self, target: CombinedPortfolioTarget) -> None:
        self._targets[target.target_id] = target

    async def list_target_contributions(
        self,
        combined_target_id: uuid.UUID,
    ) -> list[EngineTargetContribution]:
        target = self._targets.get(combined_target_id)
        return list(target.contributions) if target else []

    async def save_order_allocations(self, allocations: list[OrderAllocation]) -> None:
        for allocation in allocations:
            rows = self._allocations.setdefault(allocation.order_id, [])
            if not any(existing.allocation_id == allocation.allocation_id for existing in rows):
                rows.append(allocation)

    async def list_order_allocations(self, order_id: uuid.UUID) -> list[OrderAllocation]:
        return list(self._allocations.get(order_id, []))


class PostgresMultiEngineGovernanceRepository:
    """PostgreSQL-backed multi-engine governance repository."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save_engine_budget(self, budget: EngineBudget) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO engine_budgets
                        (engine_name, engine_version, run_mode, capital_weight,
                         max_gross, max_turnover, enabled)
                    VALUES
                        (:engine_name, :engine_version, :run_mode, :capital_weight,
                         :max_gross, :max_turnover, :enabled)
                    ON CONFLICT (engine_name)
                    DO UPDATE SET
                        engine_version = EXCLUDED.engine_version,
                        run_mode = EXCLUDED.run_mode,
                        capital_weight = EXCLUDED.capital_weight,
                        max_gross = EXCLUDED.max_gross,
                        max_turnover = EXCLUDED.max_turnover,
                        enabled = EXCLUDED.enabled,
                        updated_at = now()
                """),
                _budget_params(budget),
            )

    async def list_engine_budgets(self) -> list[EngineBudget]:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT engine_name, engine_version, run_mode,
                                   capital_weight, max_gross, max_turnover, enabled
                            FROM engine_budgets
                            ORDER BY engine_name
                        """)
                    )
                )
                .mappings()
                .all()
            )
        return [
            EngineBudget(
                engine_name=str(row["engine_name"]),
                engine_version=str(row["engine_version"]),
                run_mode=str(row["run_mode"]),
                capital_weight=Decimal(str(row["capital_weight"])),
                max_gross=Decimal(str(row["max_gross"])),
                max_turnover=Decimal(str(row["max_turnover"])),
                enabled=bool(row["enabled"]),
            )
            for row in rows
        ]

    async def save_combined_target(self, target: CombinedPortfolioTarget) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO combined_portfolio_targets
                        (target_id, as_of, weights_json, cash_target_weight,
                         construction_notes)
                    VALUES
                        (:target_id, :as_of, CAST(:weights_json AS JSONB),
                         :cash_target_weight, CAST(:construction_notes AS JSONB))
                    ON CONFLICT (target_id) DO NOTHING
                """),
                {
                    "target_id": target.target_id,
                    "as_of": target.as_of,
                    "weights_json": _weights_to_json(target.weights),
                    "cash_target_weight": target.cash_target_weight,
                    "construction_notes": json.dumps(list(target.construction_notes)),
                },
            )
            for contribution in target.contributions:
                await conn.execute(
                    text("""
                        INSERT INTO engine_target_contributions
                            (contribution_id, combined_target_id, engine_name,
                             strategy_run_id, as_of, weights_json, capital_weight)
                        VALUES
                            (:contribution_id, :combined_target_id, :engine_name,
                             :strategy_run_id, :as_of, CAST(:weights_json AS JSONB),
                             :capital_weight)
                        ON CONFLICT (contribution_id) DO NOTHING
                    """),
                    {
                        "contribution_id": contribution.contribution_id,
                        "combined_target_id": contribution.combined_target_id,
                        "engine_name": contribution.engine_name,
                        "strategy_run_id": contribution.strategy_run_id,
                        "as_of": contribution.as_of,
                        "weights_json": _weights_to_json(contribution.weights),
                        "capital_weight": contribution.capital_weight,
                    },
                )

    async def list_target_contributions(
        self,
        combined_target_id: uuid.UUID,
    ) -> list[EngineTargetContribution]:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT contribution_id, combined_target_id, engine_name,
                                   strategy_run_id, as_of, weights_json, capital_weight
                            FROM engine_target_contributions
                            WHERE combined_target_id = :combined_target_id
                            ORDER BY engine_name
                        """),
                        {"combined_target_id": combined_target_id},
                    )
                )
                .mappings()
                .all()
            )
        return [
            EngineTargetContribution(
                contribution_id=row["contribution_id"],
                combined_target_id=row["combined_target_id"],
                engine_name=str(row["engine_name"]),
                strategy_run_id=row["strategy_run_id"],
                as_of=row["as_of"],
                weights=_json_to_weights(row["weights_json"]),
                capital_weight=Decimal(str(row["capital_weight"])),
            )
            for row in rows
        ]

    async def save_order_allocations(self, allocations: list[OrderAllocation]) -> None:
        async with self._engine.begin() as conn:
            for allocation in allocations:
                await conn.execute(
                    text("""
                        INSERT INTO order_allocations
                            (allocation_id, order_id, engine_name, strategy_run_id,
                             instrument_id, allocated_weight, allocated_notional)
                        VALUES
                            (:allocation_id, :order_id, :engine_name, :strategy_run_id,
                             :instrument_id, :allocated_weight, :allocated_notional)
                        ON CONFLICT (allocation_id) DO NOTHING
                    """),
                    {
                        "allocation_id": allocation.allocation_id,
                        "order_id": allocation.order_id,
                        "engine_name": allocation.engine_name,
                        "strategy_run_id": allocation.strategy_run_id,
                        "instrument_id": allocation.instrument_id,
                        "allocated_weight": allocation.allocated_weight,
                        "allocated_notional": allocation.allocated_notional,
                    },
                )

    async def list_order_allocations(self, order_id: uuid.UUID) -> list[OrderAllocation]:
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT allocation_id, order_id, engine_name, strategy_run_id,
                                   instrument_id, allocated_weight, allocated_notional
                            FROM order_allocations
                            WHERE order_id = :order_id
                            ORDER BY engine_name
                        """),
                        {"order_id": order_id},
                    )
                )
                .mappings()
                .all()
            )
        return [
            OrderAllocation(
                allocation_id=row["allocation_id"],
                order_id=row["order_id"],
                engine_name=str(row["engine_name"]),
                strategy_run_id=row["strategy_run_id"],
                instrument_id=row["instrument_id"],
                allocated_weight=Decimal(str(row["allocated_weight"])),
                allocated_notional=Decimal(str(row["allocated_notional"]))
                if row["allocated_notional"] is not None
                else None,
            )
            for row in rows
        ]


def build_multi_engine_governance_repository(
    dsn: str,
) -> InMemoryMultiEngineGovernanceRepository | PostgresMultiEngineGovernanceRepository:
    if not dsn:
        return InMemoryMultiEngineGovernanceRepository()
    from quant_platform.infrastructure.postgres.repositories import create_pg_engine

    return PostgresMultiEngineGovernanceRepository(create_pg_engine(dsn))


def _budget_params(budget: EngineBudget) -> dict[str, object]:
    return {
        "engine_name": budget.engine_name,
        "engine_version": budget.engine_version,
        "run_mode": budget.run_mode,
        "capital_weight": budget.capital_weight,
        "max_gross": budget.max_gross,
        "max_turnover": budget.max_turnover,
        "enabled": budget.enabled,
    }


def _weights_to_json(weights: object) -> str:
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping")
    return json.dumps({str(k): str(v) for k, v in weights.items()})


def _json_to_weights(raw: object) -> dict[uuid.UUID, Decimal]:
    payload = raw if isinstance(raw, dict) else json.loads(str(raw))
    return {uuid.UUID(str(k)): Decimal(str(v)) for k, v in payload.items()}
