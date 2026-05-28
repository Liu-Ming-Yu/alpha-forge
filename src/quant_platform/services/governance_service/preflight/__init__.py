"""Production preflight checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from quant_platform.core.contracts.common import BrokerHealth, BrokerHealthStatus
from quant_platform.core.domain.production import (
    PreflightCheck,
    PreflightReport,
    ProductionProfile,
)
from quant_platform.services.governance_service.preflight.preflight_config_checks import (
    build_configuration_preflight_checks,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.market_data import MarketBar


class BrokerHealthChecker(Protocol):
    """Minimal broker surface required by ``check_broker_connectivity``."""

    async def health_check(self) -> BrokerHealth: ...


class DataFreshnessProvider(Protocol):
    """Minimal data-provider surface required by ``check_data_freshness``."""

    async def get_last_bar(
        self, instrument_id: uuid.UUID, bar_seconds: int
    ) -> MarketBar | None: ...


def evaluate_preflight(
    settings: PlatformSettings,
    *,
    profile: ProductionProfile,
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
) -> PreflightReport:
    """Return production-readiness checks for a deployment profile."""
    contracts = instrument_contracts or {}
    return PreflightReport(
        profile=profile,
        generated_at=datetime.now(tz=UTC),
        checks=build_configuration_preflight_checks(
            settings,
            profile=profile,
            instrument_contracts=contracts,
        ),
    )


def assert_preflight_passed(report: PreflightReport, profile: ProductionProfile) -> None:
    """Raise RuntimeError if any error-severity check failed for the given profile."""
    failures = [c for c in report.checks if not c.passed and c.severity == "error"]
    if failures:
        lines = "\n".join(f"  - {c.name}: {c.detail}" for c in failures)
        raise RuntimeError(
            f"Preflight failed for {profile.value} profile ({len(failures)} error(s)):\n{lines}"
        )


async def check_broker_connectivity(broker: BrokerHealthChecker) -> PreflightCheck:
    """Live broker connectivity preflight check.

    Calls ``broker.health_check()`` and returns a PreflightCheck.  Fails
    closed on any exception (treats errors as disconnected).
    """
    try:
        health = await broker.health_check()
        connected = health.status == BrokerHealthStatus.CONNECTED
        detail = f"status={health.status.value} latency={health.latency_ms:.1f}ms"
        return PreflightCheck(
            name="broker_connectivity",
            passed=connected,
            detail=detail,
            severity="error",
        )
    except Exception as exc:
        return PreflightCheck(
            name="broker_connectivity",
            passed=False,
            detail=f"broker.health_check() raised: {exc}",
            severity="error",
        )


async def check_data_freshness(
    data_provider: DataFreshnessProvider,
    instrument_id: uuid.UUID,
    bar_seconds: int,
    max_age_minutes: int,
) -> PreflightCheck:
    """Data freshness preflight check.

    Calls ``data_provider.get_last_bar()`` and fails if the bar timestamp
    is older than ``max_age_minutes``.  A missing bar also fails.
    """
    try:
        bar = await data_provider.get_last_bar(instrument_id, bar_seconds)
    except Exception as exc:
        return PreflightCheck(
            name="data_freshness",
            passed=False,
            detail=f"get_last_bar() raised: {exc}",
            severity="error",
        )
    if bar is None:
        return PreflightCheck(
            name="data_freshness",
            passed=False,
            detail=f"no bar available for reference instrument {instrument_id}",
            severity="error",
        )
    now = datetime.now(tz=UTC)
    bar_ts: datetime = (
        bar.timestamp.astimezone(UTC) if bar.timestamp.tzinfo else bar.timestamp.replace(tzinfo=UTC)
    )
    age = now - bar_ts
    max_age = timedelta(minutes=max_age_minutes)
    fresh = age <= max_age
    return PreflightCheck(
        name="data_freshness",
        passed=fresh,
        detail=(
            f"last bar timestamp={bar_ts.isoformat()} age={age.total_seconds() / 60:.1f}min "
            f"(max={max_age_minutes}min)"
        ),
        severity="error",
    )
