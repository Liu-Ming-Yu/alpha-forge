"""In-memory V2 model-artifact repository."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import (
    AlphaReadinessReport,
    ModelArtifact,
    ModelCard,
    PromotionState,
)

if TYPE_CHECKING:
    from datetime import datetime


class InMemoryModelArtifactRepository:
    """In-memory model artifact, model-card, and alpha readiness repository."""

    def __init__(self) -> None:
        self._artifacts: dict[uuid.UUID, ModelArtifact] = {}
        self._cards: dict[uuid.UUID, ModelCard] = {}
        self._reports: dict[str, list[AlphaReadinessReport]] = defaultdict(list)

    async def register_artifact(self, artifact: ModelArtifact) -> None:
        self._artifacts[artifact.artifact_id] = artifact

    async def get_artifact(self, artifact_id: uuid.UUID) -> ModelArtifact | None:
        return self._artifacts.get(artifact_id)

    async def save_model_card(self, card: ModelCard) -> None:
        self._cards[card.card_id] = card

    async def save_alpha_readiness(self, report: AlphaReadinessReport) -> None:
        rows = self._reports[report.alpha_source]
        rows[:] = [existing for existing in rows if existing.report_id != report.report_id]
        rows.append(report)
        rows.sort(key=lambda row: row.as_of)

    async def evaluate_alpha(
        self,
        source_name: str,
        *,
        as_of: datetime,
    ) -> AlphaReadinessReport:
        candidates = [row for row in self._reports.get(source_name, []) if row.as_of <= as_of]
        if not candidates:
            return AlphaReadinessReport(
                report_id=uuid.uuid4(),
                alpha_source=source_name,
                as_of=as_of,
                promotion_state=PromotionState.SHADOW,
                passed=False,
                metrics={},
                drift={},
                rollback_target="",
            )
        return candidates[-1]
