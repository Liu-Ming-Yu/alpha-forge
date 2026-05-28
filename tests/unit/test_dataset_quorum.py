"""Tests for dataset vendor-quorum evidence (R-DAT-04)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.maintenance.dataset_quorum import (
    compute_dataset_quorum_evidence,
)


def _bar(close: Decimal, *, instrument: uuid.UUID, day: int = 1) -> MarketBar:
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument,
        timestamp=datetime(2026, 1, day, tzinfo=UTC),
        bar_seconds=86400,
        open=close,
        high=close + Decimal("0.5"),
        low=close - Decimal("0.5"),
        close=close,
        volume=1_000_000,
    )


def test_quorum_passes_when_two_vendors_agree() -> None:
    instrument = uuid.uuid4()
    primary = [_bar(Decimal("100.0"), instrument=instrument)]
    secondary = [_bar(Decimal("100.05"), instrument=instrument)]

    evidence = compute_dataset_quorum_evidence(
        {"ib": primary, "tiingo": secondary},
        dataset_kind="bars_eod",
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        max_disagreement_bps=Decimal("50"),
    )

    assert evidence.passed is True
    assert evidence.vendors == ("ib", "tiingo")
    assert evidence.details["overlap_count"] == 1
    assert evidence.details["breach_count"] == 0


def test_quorum_fails_on_large_disagreement() -> None:
    instrument = uuid.uuid4()
    primary = [_bar(Decimal("100.0"), instrument=instrument)]
    secondary = [_bar(Decimal("110.0"), instrument=instrument)]

    evidence = compute_dataset_quorum_evidence(
        {"ib": primary, "tiingo": secondary},
        dataset_kind="bars_eod",
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        max_disagreement_bps=Decimal("50"),
    )

    assert evidence.passed is False
    assert evidence.details["breach_count"] == 1
    assert evidence.details["max_disagreement_bps"] > 100.0


def test_quorum_fails_when_only_one_vendor_supplies_data() -> None:
    instrument = uuid.uuid4()
    evidence = compute_dataset_quorum_evidence(
        {"ib": [_bar(Decimal("100.0"), instrument=instrument)]},
        dataset_kind="bars_eod",
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert evidence.passed is False
    assert evidence.vendors == ("ib",)


def test_quorum_fails_when_secondary_has_zero_overlap() -> None:
    instrument = uuid.uuid4()
    primary = [_bar(Decimal("100.0"), instrument=instrument, day=1)]
    secondary = [_bar(Decimal("100.0"), instrument=instrument, day=2)]

    evidence = compute_dataset_quorum_evidence(
        {"ib": primary, "tiingo": secondary},
        dataset_kind="bars_eod",
        as_of=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert evidence.passed is False
    assert evidence.details["overlap_count"] == 0
    assert evidence.details["coverage"]["ib"] < 1.0


def test_quorum_requires_three_vendors_when_configured() -> None:
    instrument = uuid.uuid4()
    bars = [_bar(Decimal("100.0"), instrument=instrument)]
    evidence = compute_dataset_quorum_evidence(
        {"ib": bars, "tiingo": list(bars)},
        dataset_kind="bars_eod",
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        required_vendor_count=3,
    )

    assert evidence.passed is False


def test_quorum_coerces_naive_as_of_to_utc() -> None:
    evidence = compute_dataset_quorum_evidence(
        {"ib": [], "tiingo": []},
        dataset_kind="bars_eod",
        as_of=datetime(2026, 1, 2),
    )
    assert evidence.as_of.tzinfo is not None
    # No bars and required_vendor_count=2: vendor count satisfied,
    # zero overlap is permitted because there is also nothing to disagree
    # on; tighter checks happen at the gate when configured for live.
    assert evidence.vendors == ("ib", "tiingo")


def test_quorum_rejects_duplicate_vendor_names() -> None:
    with pytest.raises(ValueError):
        compute_dataset_quorum_evidence(
            {"ib": [], "ib ": []},
            dataset_kind="bars_eod",
            as_of=datetime(2026, 1, 2, tzinfo=UTC),
        )
