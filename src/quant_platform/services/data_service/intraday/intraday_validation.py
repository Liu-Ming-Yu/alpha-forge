"""Validation helpers for vendor-neutral intraday bar imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.core.domain.market_data import INTRADAY_BAR_SECONDS as INTRADAY_BAR_SECONDS

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.domain.market_data import VendorBarBatch


@dataclass(frozen=True)
class IntradayValidationIssue:
    """One validation issue found in an intraday vendor batch."""

    severity: str
    code: str
    detail: str


@dataclass(frozen=True)
class IntradayValidationReport:
    """Validation summary for one imported intraday vendor batch."""

    vendor: str
    source_uri: str
    bar_seconds: int
    row_count: int
    instrument_count: int
    start_at: datetime | None
    end_at: datetime | None
    issues: tuple[IntradayValidationIssue, ...] = ()
    coverage: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(issue.severity != "error" for issue in self.issues)


def validate_vendor_bar_batch(
    batch: VendorBarBatch,
    *,
    expected_instruments: set[uuid.UUID] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> IntradayValidationReport:
    """Validate canonical 1-minute OHLCV data before production use."""
    issues: list[IntradayValidationIssue] = []
    if batch.bar_seconds != INTRADAY_BAR_SECONDS:
        issues.append(
            IntradayValidationIssue(
                "error",
                "bar_seconds_not_1m",
                f"expected 60, got {batch.bar_seconds}",
            )
        )
    if not batch.bars:
        issues.append(IntradayValidationIssue("error", "empty_batch", "no bars supplied"))

    seen: set[tuple[uuid.UUID, datetime]] = set()
    for bar in batch.bars:
        key = (bar.instrument_id, ensure_utc(bar.timestamp))
        if key in seen:
            issues.append(
                IntradayValidationIssue(
                    "error",
                    "duplicate_bar",
                    f"{bar.instrument_id} {bar.timestamp.isoformat()} appears more than once",
                )
            )
            break
        seen.add(key)
        if start is not None and ensure_utc(bar.timestamp) < ensure_utc(start):
            issues.append(
                IntradayValidationIssue("error", "bar_before_start", bar.timestamp.isoformat())
            )
            break
        if end is not None and ensure_utc(bar.timestamp) > ensure_utc(end):
            issues.append(
                IntradayValidationIssue("error", "bar_after_end", bar.timestamp.isoformat())
            )
            break

    observed = {bar.instrument_id for bar in batch.bars}
    if expected_instruments:
        missing = expected_instruments - observed
        if missing:
            issues.append(
                IntradayValidationIssue(
                    "error",
                    "missing_instrument_coverage",
                    f"missing {len(missing)} requested instruments",
                )
            )

    if batch.bars:
        start_at = min(ensure_utc(bar.timestamp) for bar in batch.bars)
        end_at = max(ensure_utc(bar.timestamp) for bar in batch.bars)
    else:
        start_at = None
        end_at = None

    return IntradayValidationReport(
        vendor=batch.vendor,
        source_uri=batch.source_uri,
        bar_seconds=batch.bar_seconds,
        row_count=len(batch.bars),
        instrument_count=len(observed),
        start_at=start_at,
        end_at=end_at,
        issues=tuple(issues),
        coverage=dict(batch.coverage),
    )


def validation_payload(report: IntradayValidationReport) -> dict[str, object]:
    """Return JSON-safe validation payload."""
    return {
        "vendor": report.vendor,
        "source_uri": report.source_uri,
        "bar_seconds": report.bar_seconds,
        "row_count": report.row_count,
        "instrument_count": report.instrument_count,
        "start_at": report.start_at.isoformat() if report.start_at else None,
        "end_at": report.end_at.isoformat() if report.end_at else None,
        "passed": report.passed,
        "coverage": report.coverage,
        "issues": [
            {"severity": issue.severity, "code": issue.code, "detail": issue.detail}
            for issue in report.issues
        ],
    }


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "INTRADAY_BAR_SECONDS",
    "IntradayValidationIssue",
    "IntradayValidationReport",
    "ensure_utc",
    "validate_vendor_bar_batch",
    "validation_payload",
]
