"""Machine-owned paper-soak evidence generator."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from quant_platform.services.governance_service.gates.signal_gate import signal_gate_status
from quant_platform.services.governance_service.paper_soak.paper_soak_runtime import (
    order_latency_section,
    reconciliation_section,
)
from quant_platform.services.governance_service.paper_soak.paper_soak_sections import (
    broker_health_section,
    data_health_section,
    lifecycle_section,
    nav_section,
    signal_gate_section,
)
from quant_platform.services.governance_service.support.prediction_evidence import (
    forecast_evidence_payload,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import SignalPromotionGate
    from quant_platform.core.domain.production import (
        BrokerHealthObservation,
        BrokerSmokeObservation,
        DataHealthReport,
        ForecastEvidence,
        NavSnapshot,
        PaperLifecycleObservation,
        SignalGateStatus,
    )

SOAK_REPORT_VERSION = 1


class PerformanceRepoLike(Protocol):
    """Minimal repository surface used by ``build_paper_soak_report``."""

    async def latest_broker_health(self) -> BrokerHealthObservation | None: ...
    async def latest_broker_smoke(self) -> BrokerSmokeObservation | None: ...
    async def latest_paper_lifecycle(self) -> PaperLifecycleObservation | None: ...
    async def list_nav_snapshots(
        self, strategy_run_id: uuid.UUID, *, limit: int = 252
    ) -> list[NavSnapshot]: ...
    async def forecast_evidence(
        self,
        source: str,
        *,
        model_version: str | None = None,
        as_of: datetime,
        stale_after_hours: int = 24,
        min_confidence: float = 0.0,
        limit: int = 500,
    ) -> ForecastEvidence: ...


DataHealthEvidenceBuilder = Callable[
    [Any, datetime, Path | str | None, int, int],
    Awaitable[tuple[Any | None, str]],
]


def build_performance_repository(_dsn: str | None) -> PerformanceRepoLike:
    """Compatibility injection hook for tests; bootstrap supplies production repositories."""
    raise RuntimeError("paper-soak repository must be supplied by bootstrap")


async def _reconciliation_section(settings: PlatformSettings) -> dict[str, Any]:
    """Compatibility wrapper for durable reconciliation evidence."""
    return await reconciliation_section(settings)


async def _order_latency_section(
    settings: PlatformSettings,
    *,
    as_of: datetime,
    window_days: int,
) -> dict[str, Any]:
    """Compatibility wrapper for durable order-latency evidence."""
    return await order_latency_section(settings, as_of=as_of, window_days=window_days)


async def _build_data_health_evidence(
    settings: PlatformSettings,
    *,
    as_of: datetime,
    contracts_file: Path | str | None,
    bar_seconds: int,
    data_health_window_days: int,
    data_health_builder: DataHealthEvidenceBuilder | None = None,
) -> tuple[DataHealthReport | None, str]:
    """Build data-health evidence when a contracts file is supplied."""
    if contracts_file is None:
        return None, "no contracts file supplied"
    if data_health_builder is None:
        return None, "data-health builder is not configured"

    try:
        return await data_health_builder(
            settings,
            as_of,
            contracts_file,
            bar_seconds,
            data_health_window_days,
        )
    except Exception as exc:
        return None, f"data-health build failed: {exc}"


async def _signal_gate_evidence(
    settings: PlatformSettings,
    *,
    signal_name: str,
    signal_type: str,
    as_of: datetime,
    gate: SignalPromotionGate | None = None,
) -> SignalGateStatus | None:
    """Load signal-gate evidence when a signal name is supplied."""
    if not signal_name:
        return None
    try:
        kwargs: dict[str, Any] = {
            "signal_name": signal_name,
            "signal_type": signal_type,
            "as_of": as_of,
        }
        if gate is not None:
            kwargs["gate"] = gate
        return await signal_gate_status(settings, **kwargs)
    except Exception:
        return None


def _broker_smoke_section(smoke: BrokerSmokeObservation) -> dict[str, Any]:
    return {
        "observed_at": smoke.observed_at.isoformat(),
        "status": smoke.status,
        "host": smoke.host,
        "port": smoke.port,
        "client_id": smoke.client_id,
        "latency_ms": smoke.latency_ms,
        "account_status": smoke.account_status,
        "positions_status": smoke.positions_status,
        "open_orders_status": smoke.open_orders_status,
        "passed": smoke.passed,
    }


async def build_paper_soak_report(
    settings: PlatformSettings,
    *,
    as_of: datetime,
    strategy_run_id: uuid.UUID,
    contracts_file: Path | str | None = None,
    signal_name: str = "",
    signal_type: str = "classical",
    bar_seconds: int = 86400,
    data_health_window_days: int = 5,
    order_latency_window_days: int = 7,
    performance_repository: PerformanceRepoLike | None = None,
    data_health_builder: DataHealthEvidenceBuilder | None = None,
    signal_gate: SignalPromotionGate | None = None,
    reconciliation_evidence: dict[str, Any] | None = None,
    order_latency_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a machine-owned paper-soak report payload."""
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)

    repo = performance_repository or build_performance_repository(settings.storage.postgres_dsn)

    stale_after = timedelta(days=max(1, settings.production.data_health_stale_after_days))
    health = await repo.latest_broker_health()
    smoke = await repo.latest_broker_smoke()
    lifecycle = await repo.latest_paper_lifecycle()
    navs = await repo.list_nav_snapshots(strategy_run_id, limit=1)
    latest_nav = navs[-1] if navs else None

    data_report, data_health_error = await _build_data_health_evidence(
        settings,
        as_of=as_of,
        contracts_file=contracts_file,
        bar_seconds=bar_seconds,
        data_health_window_days=data_health_window_days,
        data_health_builder=data_health_builder,
    )
    signal_status = await _signal_gate_evidence(
        settings,
        signal_name=signal_name,
        signal_type=signal_type,
        as_of=as_of,
        gate=signal_gate,
    )
    prediction_sources = [
        source
        for source, weight in settings.alpha.source_weights.items()
        if source != "classical" and float(weight) > 0
    ]
    prediction_quality = []
    for source in prediction_sources:
        prediction_quality.append(
            forecast_evidence_payload(
                await repo.forecast_evidence(
                    source,
                    as_of=as_of,
                    stale_after_hours=settings.production.prediction_evidence_stale_after_hours,
                    min_confidence=settings.production.prediction_evidence_min_confidence,
                )
            )
        )

    payload: dict[str, Any] = {
        "version": SOAK_REPORT_VERSION,
        "generated_at": as_of.astimezone(UTC).isoformat(),
        "strategy_run_id": str(strategy_run_id),
        "broker_health": broker_health_section(health, as_of=as_of, stale_after=stale_after),
        "lifecycle_result": lifecycle_section(lifecycle, as_of=as_of, stale_after=stale_after),
        "nav_snapshot": nav_section(latest_nav),
        "data_health": (
            data_health_section(data_report)
            if data_report is not None
            else {"passed": False, "detail": data_health_error}
        ),
        "signal_gate": signal_gate_section(signal_status),
        "prediction_quality": prediction_quality,
        "reconciliation": reconciliation_evidence
        if reconciliation_evidence is not None
        else await _reconciliation_section(settings),
        "order_latency": order_latency_evidence
        if order_latency_evidence is not None
        else await _order_latency_section(
            settings,
            as_of=as_of,
            window_days=order_latency_window_days,
        ),
    }
    if smoke is not None:
        payload["broker_smoke"] = _broker_smoke_section(smoke)
    return payload


def write_paper_soak_report(payload: dict[str, Any], output: Path) -> Path:
    """Persist the soak report to disk and return the canonical path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output
