"""In-memory performance repository adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.infrastructure.performance.evidence_inmemory import (
    InMemoryPredictionEvidenceMixin,
    InMemoryShadowPaperParityMixin,
)
from quant_platform.infrastructure.performance.inmemory_rows import (
    append_if_missing_sorted,
    append_sorted,
    latest,
    upsert_sorted,
)
from quant_platform.infrastructure.performance.signal_inmemory import InMemorySignalGateMixin
from quant_platform.infrastructure.performance.status import build_performance_report

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.production import (
        BrokerHealthObservation,
        BrokerSmokeObservation,
        InstrumentPnl,
        MetricRollupSnapshot,
        NavSnapshot,
        PaperLifecycleObservation,
        PerformanceReport,
        PredictionResult,
        RuntimeHeartbeat,
        ShadowPaperParityRecord,
        SignalGateRecord,
        TextSignalGateRecord,
    )


class InMemoryPerformanceRepository(
    InMemorySignalGateMixin,
    InMemoryPredictionEvidenceMixin,
    InMemoryShadowPaperParityMixin,
):
    """In-memory performance repository for tests and local paper runs."""

    def __init__(self) -> None:
        self._nav: dict[uuid.UUID, list[NavSnapshot]] = {}
        self._ic: dict[str, list[TextSignalGateRecord]] = {}
        self._signals: dict[tuple[str, str], list[SignalGateRecord]] = {}
        self._heartbeats: dict[str, RuntimeHeartbeat] = {}
        self._broker_health: list[BrokerHealthObservation] = []
        self._broker_smoke: list[BrokerSmokeObservation] = []
        self._paper_lifecycle: list[PaperLifecycleObservation] = []
        self._instrument_pnl: list[InstrumentPnl] = []
        self._predictions: list[PredictionResult] = []
        self._metric_rollups: list[MetricRollupSnapshot] = []
        self._shadow_paper_parity: list[ShadowPaperParityRecord] = []

    async def save_nav_snapshot(self, snapshot: NavSnapshot) -> None:
        rows = self._nav.setdefault(snapshot.strategy_run_id, [])
        append_if_missing_sorted(
            rows,
            snapshot,
            identity=lambda row: row.snapshot_id,
            sort_key=lambda row: row.as_of,
        )

    async def list_nav_snapshots(
        self,
        strategy_run_id: uuid.UUID,
        *,
        limit: int = 252,
    ) -> list[NavSnapshot]:
        rows = list(self._nav.get(strategy_run_id, []))
        return rows[-limit:]

    async def performance_report(
        self,
        strategy_run_id: uuid.UUID,
        *,
        as_of: datetime,
        window: int = 90,
    ) -> PerformanceReport:
        rows = await self.list_nav_snapshots(strategy_run_id, limit=window + 1)
        return build_performance_report(strategy_run_id, as_of=as_of, rows=rows)

    async def save_metric_rollup(self, snapshot: MetricRollupSnapshot) -> None:
        upsert_sorted(
            self._metric_rollups,
            snapshot,
            identity=lambda row: row.snapshot_id,
            sort_key=lambda row: row.as_of,
        )

    async def list_metric_rollups(
        self,
        metric_name: str | None = None,
        *,
        limit: int = 500,
    ) -> list[MetricRollupSnapshot]:
        rows = list(self._metric_rollups)
        if metric_name is not None:
            rows = [row for row in rows if row.metric_name == metric_name]
        return rows[-limit:]

    async def save_runtime_heartbeat(self, heartbeat: RuntimeHeartbeat) -> None:
        self._heartbeats[heartbeat.component] = heartbeat

    async def latest_runtime_heartbeat(self, component: str) -> RuntimeHeartbeat | None:
        return self._heartbeats.get(component)

    async def save_broker_health(self, observation: BrokerHealthObservation) -> None:
        append_sorted(self._broker_health, observation, sort_key=lambda row: row.observed_at)

    async def latest_broker_health(self) -> BrokerHealthObservation | None:
        return latest(self._broker_health)

    async def save_broker_smoke(self, observation: BrokerSmokeObservation) -> None:
        append_sorted(self._broker_smoke, observation, sort_key=lambda row: row.observed_at)

    async def latest_broker_smoke(self) -> BrokerSmokeObservation | None:
        return latest(self._broker_smoke)

    async def save_paper_lifecycle(self, observation: PaperLifecycleObservation) -> None:
        append_sorted(self._paper_lifecycle, observation, sort_key=lambda row: row.observed_at)

    async def latest_paper_lifecycle(self) -> PaperLifecycleObservation | None:
        return latest(self._paper_lifecycle)

    async def record_instrument_pnl(self, record: InstrumentPnl) -> None:
        self._instrument_pnl.append(record)

    async def list_instrument_pnl(
        self,
        strategy_run_id: uuid.UUID,
        *,
        limit: int = 252,
    ) -> list[InstrumentPnl]:
        rows = [r for r in self._instrument_pnl if r.strategy_run_id == strategy_run_id]
        rows.sort(key=lambda r: r.as_of)
        return rows[-limit:]
