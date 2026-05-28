"""Concrete paper-soak evidence wiring for bootstrap callers."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.bootstrap.data.instrument_contracts import load_instrument_contracts
from quant_platform.bootstrap.persistence.execution_stores import build_kill_switch_store
from quant_platform.bootstrap.persistence.postgres import create_pg_engine
from quant_platform.bootstrap.session.public_api import create_paper_session
from quant_platform.services.governance_service.gates.data_health import build_data_health_report
from quant_platform.services.governance_service.paper_soak.paper_soak_runtime import (
    order_latency_section as service_order_latency_section,
)
from quant_platform.services.governance_service.paper_soak.paper_soak_runtime import (
    reconciliation_section as service_reconciliation_section,
)
from quant_platform.services.governance_service.paper_soak.paper_soak_sections import midnight_utc

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.production import DataHealthReport


async def build_data_health_evidence(
    settings: PlatformSettings,
    as_of: datetime,
    contracts_file: Path | str | None,
    bar_seconds: int,
    data_health_window_days: int,
) -> tuple[DataHealthReport | None, str]:
    """Build paper-soak data-health evidence from the configured runtime session."""
    if contracts_file is None:
        return None, "no contracts file supplied"
    contracts = load_instrument_contracts(contracts_file)
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("0"),
        instrument_contracts=contracts,
    )
    window_end = midnight_utc(as_of)
    window_start = window_end - timedelta(days=max(1, data_health_window_days))
    report = await build_data_health_report(
        instruments=session.contract_master.list_active(),
        bar_store=session.bar_store,
        universe_manager=session.universe_manager,
        start=window_start,
        end=window_end,
        bar_seconds=bar_seconds,
        stale_after_days=settings.production.data_health_stale_after_days,
    )
    return report, ""


async def reconciliation_section(settings: PlatformSettings) -> dict[str, object]:
    """Build durable reconciliation evidence using concrete stores."""
    return await service_reconciliation_section(
        settings,
        kill_switch_store_factory=build_kill_switch_store,
    )


async def order_latency_section(
    settings: PlatformSettings,
    *,
    as_of: datetime,
    window_days: int,
) -> dict[str, object]:
    """Build durable order-latency evidence using the configured Postgres engine."""
    return await service_order_latency_section(
        settings,
        as_of=as_of,
        window_days=window_days,
        pg_engine_factory=create_pg_engine,
    )


__all__ = [
    "build_data_health_evidence",
    "order_latency_section",
    "reconciliation_section",
]
