"""Dataset vendor-quorum evidence (R-DAT-04).

Computes :class:`DatasetQuorumEvidence` from the per-instrument output
of two or more independent vendor adapters.  The evidence captures:

- the set of vendors consulted;
- whether each vendor returned coverage for every (instrument, date)
  observed by any other vendor;
- per-bar disagreement in basis points (close prices) and the maximum
  disagreement across all overlapping bars;
- the configured ``required_vendor_count`` and ``max_disagreement_bps``
  thresholds applied to derive the ``passed`` flag.

The evidence is persisted via :class:`DatasetCatalog.record_quorum_evidence`
so the readiness and ``production-candidate`` gates can require fresh
persisted evidence rather than configuration flags alone.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

from quant_platform.core.domain.market_data import DatasetQuorumEvidence, MarketBar

VendorBars = Mapping[str, Iterable[MarketBar]]


@dataclass(frozen=True)
class _BarKey:
    instrument_id: object
    bar_date: date


def _bar_key(bar: MarketBar) -> _BarKey:
    return _BarKey(instrument_id=bar.instrument_id, bar_date=bar.timestamp.date())


def _abs_disagreement_bps(a: Decimal, b: Decimal) -> Decimal:
    """Absolute disagreement between two close prices in basis points."""
    if a <= 0 and b <= 0:
        return Decimal("0")
    base = (a + b) / Decimal("2") if a > 0 and b > 0 else (a if a > 0 else b)
    if base == 0:
        return Decimal("0")
    return (abs(a - b) / base) * Decimal("10000")


def compute_dataset_quorum_evidence(
    vendor_bars: VendorBars,
    *,
    dataset_kind: str,
    as_of: datetime,
    required_vendor_count: int = 2,
    max_disagreement_bps: Decimal = Decimal("50"),
    evidence_id: uuid.UUID | None = None,
) -> DatasetQuorumEvidence:
    """Compute quorum evidence from per-vendor bar batches.

    Args:
        vendor_bars: Map from vendor name to its returned bars.  At least
            two vendors must contribute non-empty results for the
            evidence to ``pass``.
        dataset_kind: Identifier persisted with the evidence
            (e.g. ``"bars_eod"``).
        as_of: Timestamp the evidence applies to (must be tz-aware).
        required_vendor_count: Minimum vendor count for ``passed``.
        max_disagreement_bps: Per-bar tolerance in basis points.

    Returns:
        :class:`DatasetQuorumEvidence` with ``details`` describing per-vendor
        bar counts, the number of overlapping bars, the maximum disagreement
        observed, and the count of bars exceeding the tolerance.

    Raises:
        ValueError: If ``vendor_bars`` is empty, if ``as_of`` is naive,
            or if vendor names are duplicated/empty.
    """
    if not vendor_bars:
        raise ValueError("vendor_bars must be non-empty")
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)

    cleaned_vendors: dict[str, list[MarketBar]] = {}
    for vendor, bars in vendor_bars.items():
        name = vendor.strip()
        if not name:
            raise ValueError("vendor names must not be empty")
        if name in cleaned_vendors:
            raise ValueError(f"duplicate vendor: {name!r}")
        cleaned_vendors[name] = list(bars)

    vendor_keys: dict[str, dict[_BarKey, MarketBar]] = {
        vendor: {_bar_key(bar): bar for bar in bars} for vendor, bars in cleaned_vendors.items()
    }

    union_keys: set[_BarKey] = set()
    for keys in vendor_keys.values():
        union_keys.update(keys.keys())

    coverage = {
        vendor: len(keys) / len(union_keys) if union_keys else 1.0
        for vendor, keys in vendor_keys.items()
    }
    bar_counts = {vendor: len(bars) for vendor, bars in cleaned_vendors.items()}

    overlap_keys: set[_BarKey] = set()
    if union_keys and len(vendor_keys) >= 2:
        # bars present in *every* vendor's output
        iter_keys = list(vendor_keys.values())
        overlap_keys = set(iter_keys[0].keys())
        for keys in iter_keys[1:]:
            overlap_keys &= set(keys.keys())

    max_bps = Decimal("0")
    breaches: list[dict[str, object]] = []
    if overlap_keys and len(vendor_keys) >= 2:
        # Reference vendor: the first one (deterministic order).
        primary_name = next(iter(vendor_keys))
        primary = vendor_keys[primary_name]
        for key in overlap_keys:
            primary_bar = primary[key]
            for other, other_keys in vendor_keys.items():
                if other == primary_name:
                    continue
                other_bar = other_keys[key]
                bps = _abs_disagreement_bps(primary_bar.close, other_bar.close)
                if bps > max_bps:
                    max_bps = bps
                if bps > max_disagreement_bps:
                    breaches.append(
                        {
                            "instrument_id": str(primary_bar.instrument_id),
                            "bar_date": key.bar_date.isoformat(),
                            "primary": primary_name,
                            "primary_close": float(primary_bar.close),
                            "secondary": other,
                            "secondary_close": float(other_bar.close),
                            "disagreement_bps": float(bps),
                        }
                    )

    enough_vendors = len(cleaned_vendors) >= required_vendor_count
    enough_overlap = bool(overlap_keys) or not union_keys
    passed = bool(
        enough_vendors and enough_overlap and not breaches and max_bps <= max_disagreement_bps
    )

    details: dict[str, object] = {
        "bar_counts": bar_counts,
        "coverage": coverage,
        "overlap_count": len(overlap_keys),
        "union_count": len(union_keys),
        "max_disagreement_bps": float(max_bps),
        "breach_count": len(breaches),
        "breach_samples": breaches[:10],
    }

    return DatasetQuorumEvidence(
        evidence_id=evidence_id or uuid.uuid4(),
        dataset_kind=dataset_kind,
        as_of=as_of,
        vendors=tuple(cleaned_vendors.keys()),
        passed=passed,
        required_vendor_count=required_vendor_count,
        max_disagreement_bps=max_disagreement_bps,
        details=details,
    )


@dataclass(frozen=True)
class QuorumStaleness:
    """Result of asserting that persisted quorum evidence is fresh."""

    fresh: bool
    detail: str
    evidence: DatasetQuorumEvidence | None = None
    extras: dict[str, object] = field(default_factory=dict)
