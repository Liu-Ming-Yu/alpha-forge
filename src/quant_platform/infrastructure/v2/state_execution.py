"""In-memory V2 execution-quality repository."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.domain.orders import ExecutionQualityReport


class InMemoryExecutionQualityRepository:
    """In-memory execution-quality report repository."""

    def __init__(self) -> None:
        self._reports: dict[uuid.UUID, list[ExecutionQualityReport]] = defaultdict(list)

    async def save_execution_quality(self, report: ExecutionQualityReport) -> None:
        rows = self._reports[report.order_id]
        rows[:] = [existing for existing in rows if existing.report_id != report.report_id]
        rows.append(report)
        rows.sort(key=lambda row: row.as_of)

    async def list_execution_quality(
        self,
        order_id: uuid.UUID,
    ) -> list[ExecutionQualityReport]:
        return list(self._reports.get(order_id, []))
