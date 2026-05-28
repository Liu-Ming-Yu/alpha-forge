"""Data-health reporting for cycle go/no-go decisions."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import (
    DataHealthInstrumentStatus,
    DataHealthReport,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.contracts import HistoricalDataStore, LiquidityProfileProvider
    from quant_platform.core.domain.instruments import Instrument


async def build_data_health_report(
    *,
    instruments: list[Instrument],
    bar_store: HistoricalDataStore,
    universe_manager: LiquidityProfileProvider,
    start: datetime,
    end: datetime,
    bar_seconds: int = 86400,
    stale_after_days: int = 3,
) -> DataHealthReport:
    """Build a data-health report for a universe and window."""
    statuses: list[DataHealthInstrumentStatus] = []
    stale_cutoff = end - timedelta(days=stale_after_days)
    for instrument in instruments:
        bars = await bar_store.get_bars(
            instrument.instrument_id,
            bar_seconds,
            start,
            end,
        )
        latest = max((bar.timestamp for bar in bars), default=None)
        profile = universe_manager.get_profile(instrument.instrument_id)
        issues: list[str] = []
        if not bars:
            issues.append("missing_bars")
        if profile is None:
            issues.append("missing_liquidity_profile")
        stale = latest is None or latest < stale_cutoff
        if stale:
            issues.append("stale_bars")
        statuses.append(
            DataHealthInstrumentStatus(
                instrument_id=instrument.instrument_id,
                symbol=instrument.symbol,
                bars_found=len(bars),
                latest_bar_at=latest,
                liquidity_profile_present=profile is not None,
                stale=stale,
                issues=tuple(issues),
            )
        )

    return DataHealthReport(
        generated_at=end,
        start=start,
        end=end,
        instruments_checked=len(statuses),
        instruments_with_bars=sum(1 for status in statuses if status.bars_found > 0),
        instruments_with_liquidity=sum(
            1 for status in statuses if status.liquidity_profile_present
        ),
        stale_instruments=sum(1 for status in statuses if status.stale),
        statuses=tuple(statuses),
    )


def data_health_payload(report: DataHealthReport) -> dict[str, object]:
    """JSON-serialisable representation for CLI and operator output."""
    return {
        "generated_at": report.generated_at.isoformat(),
        "start": report.start.isoformat(),
        "end": report.end.isoformat(),
        "passed": report.passed,
        "coverage_pct": report.coverage_pct,
        "liquidity_coverage_pct": report.liquidity_coverage_pct,
        "stale_instruments": report.stale_instruments,
        "instruments_checked": report.instruments_checked,
        "statuses": [
            {
                "instrument_id": str(status.instrument_id),
                "symbol": status.symbol,
                "bars_found": status.bars_found,
                "latest_bar_at": status.latest_bar_at.isoformat()
                if status.latest_bar_at is not None
                else None,
                "liquidity_profile_present": status.liquidity_profile_present,
                "stale": status.stale,
                "issues": list(status.issues),
            }
            for status in report.statuses
        ],
    }


def active_instrument_ids(instruments: list[Instrument]) -> set[uuid.UUID]:
    """Return active instrument IDs; kept as a tiny helper for tests/CLI."""
    return {instrument.instrument_id for instrument in instruments if instrument.active}
