"""PostgreSQL scheduled feature-job persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from quant_platform.infrastructure.postgres.retry import with_retry as _with_retry
from quant_platform.services.research_service.modeling.registry.model_registry import FeatureJob

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncEngine


class PostgresFeatureJobsMixin:
    """Scheduled feature-job methods for ``PostgresModelRegistry``."""

    _engine: AsyncEngine

    async def schedule_feature_job(
        self,
        *,
        model_id: uuid.UUID,
        strategy_name: str,
        feature_set_version: str,
        interval_seconds: float,
        as_of: datetime,
    ) -> FeatureJob:
        job_id = uuid.uuid4()

        async def _do() -> None:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        """
                        INSERT INTO feature_jobs
                            (job_id, model_id, strategy_name, feature_set_version,
                             interval_seconds, next_run_at, enabled, consecutive_failures)
                        VALUES
                            (:job_id, :model_id, :strategy_name, :feature_set_version,
                             :interval_seconds, :next_run_at, true, 0)
                        """
                    ),
                    {
                        "job_id": job_id,
                        "model_id": model_id,
                        "strategy_name": strategy_name,
                        "feature_set_version": feature_set_version,
                        "interval_seconds": float(interval_seconds),
                        "next_run_at": as_of,
                    },
                )

        await _with_retry(_do)
        return FeatureJob(
            job_id=job_id,
            model_id=model_id,
            strategy_name=strategy_name,
            feature_set_version=feature_set_version,
            interval_seconds=float(interval_seconds),
            next_run_at=as_of,
        )

    async def due_feature_jobs(self, as_of: datetime) -> list[FeatureJob]:
        async def _do() -> list[FeatureJob]:
            async with self._engine.connect() as conn:
                rows = (
                    (
                        await conn.execute(
                            text(
                                """
                            SELECT fj.job_id, fj.model_id, fj.strategy_name,
                                   fj.feature_set_version, fj.interval_seconds,
                                   fj.next_run_at, fj.last_run_at, fj.enabled,
                                   fj.consecutive_failures
                            FROM feature_jobs fj
                            JOIN registered_models rm ON rm.model_id = fj.model_id
                            WHERE fj.enabled = true
                              AND fj.next_run_at <= :as_of
                              AND rm.active = true
                            ORDER BY fj.next_run_at ASC
                            """
                            ),
                            {"as_of": as_of},
                        )
                    )
                    .mappings()
                    .all()
                )
            return [_row_to_feature_job(dict(row)) for row in rows]

        return await _with_retry(_do)

    async def mark_job_completed(
        self,
        job_id: uuid.UUID,
        *,
        run_at: datetime,
        success: bool,
    ) -> None:
        async def _do() -> None:
            async with self._engine.begin() as conn:
                row = (
                    (
                        await conn.execute(
                            text(
                                """
                            SELECT interval_seconds, consecutive_failures
                            FROM feature_jobs
                            WHERE job_id = :job_id
                            FOR UPDATE
                            """
                            ),
                            {"job_id": job_id},
                        )
                    )
                    .mappings()
                    .first()
                )
                if row is None:
                    return
                interval = float(row["interval_seconds"])
                failures = int(row["consecutive_failures"])
                if success:
                    new_failures = 0
                    next_run = run_at + timedelta(seconds=interval)
                else:
                    new_failures = failures + 1
                    retry_seconds = min(300.0, max(5.0, interval / 2.0))
                    next_run = run_at + timedelta(seconds=retry_seconds)
                await conn.execute(
                    text(
                        """
                        UPDATE feature_jobs
                        SET last_run_at = :run_at,
                            next_run_at = :next_run,
                            consecutive_failures = :failures
                        WHERE job_id = :job_id
                        """
                    ),
                    {
                        "job_id": job_id,
                        "run_at": run_at,
                        "next_run": next_run,
                        "failures": new_failures,
                    },
                )

        await _with_retry(_do)


def _row_to_feature_job(row: Mapping[str, Any]) -> FeatureJob:
    return FeatureJob(
        job_id=uuid.UUID(str(row["job_id"])),
        model_id=uuid.UUID(str(row["model_id"])),
        strategy_name=row["strategy_name"],
        feature_set_version=row["feature_set_version"],
        interval_seconds=float(row["interval_seconds"]),
        next_run_at=row["next_run_at"],
        last_run_at=row["last_run_at"],
        enabled=bool(row["enabled"]),
        consecutive_failures=int(row["consecutive_failures"]),
    )
