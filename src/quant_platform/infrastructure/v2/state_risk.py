"""In-memory V2 risk-model repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.portfolio import PortfolioRiskModel, RiskSnapshot


class InMemoryPortfolioRiskModelRepository:
    """In-memory risk-model and risk-snapshot repository."""

    def __init__(self) -> None:
        self._models: list[PortfolioRiskModel] = []
        self._snapshots: dict[uuid.UUID, RiskSnapshot] = {}

    async def save_risk_model(self, model: PortfolioRiskModel) -> None:
        self._models = [
            existing for existing in self._models if existing.model_id != model.model_id
        ]
        self._models.append(model)
        self._models.sort(key=lambda row: row.as_of)

    async def latest_risk_model(self, *, as_of: datetime) -> PortfolioRiskModel | None:
        candidates = [row for row in self._models if row.as_of <= as_of]
        return candidates[-1] if candidates else None

    async def save_risk_snapshot(self, snapshot: RiskSnapshot) -> None:
        self._snapshots[snapshot.snapshot_id] = snapshot
