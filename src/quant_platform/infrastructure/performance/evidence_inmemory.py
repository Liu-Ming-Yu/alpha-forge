"""In-memory prediction and parity evidence operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.infrastructure.performance.inmemory_rows import upsert_sorted
from quant_platform.infrastructure.performance.prediction_status import build_forecast_evidence
from quant_platform.infrastructure.performance.status import build_shadow_paper_parity_status

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.production import (
        ForecastEvidence,
        PredictionResult,
        ShadowPaperParityRecord,
        ShadowPaperParityStatus,
    )


class InMemoryPredictionEvidenceMixin:
    """Prediction-evidence methods backed by in-memory rows."""

    _predictions: list[PredictionResult]

    async def save_prediction_result(self, result: PredictionResult) -> None:
        upsert_sorted(
            self._predictions,
            result,
            identity=lambda row: row.prediction_id,
            sort_key=lambda row: row.as_of,
        )

    async def list_prediction_results(
        self,
        *,
        source: str | None = None,
        model_version: str | None = None,
        strategy_run_id: uuid.UUID | None = None,
        instrument_id: uuid.UUID | None = None,
        as_of: datetime | None = None,
        limit: int = 500,
    ) -> list[PredictionResult]:
        rows = list(self._predictions)
        if source is not None:
            rows = [row for row in rows if row.source == source]
        if model_version is not None:
            rows = [row for row in rows if row.model_version == model_version]
        if strategy_run_id is not None:
            rows = [row for row in rows if row.strategy_run_id == strategy_run_id]
        if instrument_id is not None:
            rows = [row for row in rows if row.instrument_id == instrument_id]
        if as_of is not None:
            rows = [row for row in rows if row.as_of <= as_of]
        rows.sort(key=lambda row: row.as_of)
        return rows[-limit:]

    async def forecast_evidence(
        self,
        source: str,
        *,
        model_version: str | None = None,
        as_of: datetime,
        stale_after_hours: int = 24,
        min_confidence: float = 0.0,
        limit: int = 500,
    ) -> ForecastEvidence:
        rows = await self.list_prediction_results(
            source=source,
            model_version=model_version,
            as_of=as_of,
            limit=limit,
        )
        return build_forecast_evidence(
            source,
            model_version=model_version,
            as_of=as_of,
            rows=rows,
            stale_after_hours=stale_after_hours,
            min_confidence=min_confidence,
        )


class InMemoryShadowPaperParityMixin:
    """Shadow-vs-paper parity methods backed by in-memory rows."""

    _shadow_paper_parity: list[ShadowPaperParityRecord]

    async def save_shadow_paper_parity(self, record: ShadowPaperParityRecord) -> None:
        upsert_sorted(
            self._shadow_paper_parity,
            record,
            identity=lambda row: row.parity_id,
            sort_key=lambda row: (row.signal_type, row.signal_name, row.as_of),
        )

    async def list_shadow_paper_parity(
        self,
        signal_name: str,
        signal_type: str,
        *,
        as_of: datetime,
        limit: int = 252,
    ) -> list[ShadowPaperParityRecord]:
        rows = [
            row
            for row in self._shadow_paper_parity
            if row.signal_name == signal_name
            and row.signal_type == signal_type
            and row.as_of <= as_of
        ]
        rows.sort(key=lambda row: row.as_of)
        return rows[-max(1, limit) :]

    async def shadow_paper_parity_status(
        self,
        signal_name: str,
        signal_type: str,
        *,
        as_of: datetime,
        min_trading_days: int = 20,
        max_target_weight_diff_bps: float = 1.0,
        limit: int = 252,
    ) -> ShadowPaperParityStatus:
        rows = await self.list_shadow_paper_parity(
            signal_name,
            signal_type,
            as_of=as_of,
            limit=limit,
        )
        return build_shadow_paper_parity_status(
            signal_name,
            signal_type,
            as_of=as_of,
            records=rows,
            min_trading_days=min_trading_days,
            max_target_weight_diff_bps=max_target_weight_diff_bps,
        )
