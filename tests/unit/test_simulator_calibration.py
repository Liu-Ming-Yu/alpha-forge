"""Unit tests for ``simulator_calibration``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from quant_platform.services.governance_service.simulator_calibration import (
    CalibrationBucket,
    CalibrationReport,
    FillObservation,
    _adv_bucket,
    _percentile,
    calibration_payload,
    compute_calibration_report,
    write_calibration_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def _obs(
    *,
    tactic: str = "twap",
    quantity: int = 100,
    adv: float = 100_000.0,
    slippage: float = 5.0,
    spread: float = 1.0,
    side: str = "buy",
    stale_age: float = 0.0,
    order_type: str = "limit",
    when: datetime | None = None,
) -> FillObservation:
    return FillObservation(
        tactic=tactic,
        side=side,
        quantity=quantity,
        adv_shares_20d=adv,
        spread_bps=spread,
        slippage_bps=slippage,
        executed_at=when or datetime(2026, 4, 1, tzinfo=UTC),
        stale_price_age_seconds=stale_age,
        order_type=order_type,
    )


def test_adv_bucket_assignment_ranges() -> None:
    assert _adv_bucket(0.001) == "<0.5pct_adv"
    assert _adv_bucket(0.01) == "0.5-2pct_adv"
    assert _adv_bucket(0.03) == "2-5pct_adv"
    assert _adv_bucket(0.10) == ">=5pct_adv"
    assert _adv_bucket(0.0) == "<0.5pct_adv"


def test_percentile_handles_empty_and_singleton() -> None:
    assert _percentile([], 0.9) == 0.0
    assert _percentile([7.0], 0.9) == 7.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == pytest.approx(3.0)


def test_compute_report_clamps_negative_slippage_and_floor() -> None:
    obs = [
        _obs(slippage=-2.0),
        _obs(slippage=0.0),
        _obs(slippage=4.0),
    ]
    report = compute_calibration_report(
        obs,
        as_of=datetime(2026, 4, 1, tzinfo=UTC),
        floor_bps=1.0,
        min_sample_count=10,
    )
    assert report.sample_count == 3
    assert report.overall_recommended_bps >= 1.0
    assert report.insufficient_data is True
    assert report.overall_median_bps == pytest.approx(0.0)


def test_compute_report_groups_by_tactic_and_adv() -> None:
    obs = [
        _obs(tactic="twap", quantity=100, adv=100_000.0, slippage=2.0),
        _obs(tactic="twap", quantity=100, adv=100_000.0, slippage=4.0),
        _obs(tactic="vwap", quantity=10_000, adv=100_000.0, slippage=18.0),
    ]
    report = compute_calibration_report(
        obs,
        as_of=datetime(2026, 4, 1, tzinfo=UTC),
        floor_bps=1.0,
        min_sample_count=2,
    )
    assert report.insufficient_data is False
    keyed = {(b.tactic, b.adv_bucket): b for b in report.buckets}
    twap = keyed[("twap", "<0.5pct_adv")]
    vwap = keyed[("vwap", ">=5pct_adv")]
    assert twap.recommended_bps >= twap.median_slippage_bps
    assert vwap.recommended_bps >= 18.0
    assert vwap.recommended_bps > twap.recommended_bps


def test_compute_report_groups_by_execution_realism_buckets() -> None:
    obs = [
        _obs(
            spread=1.0,
            stale_age=1.0,
            order_type="limit",
            when=datetime(2026, 4, 1, 14, 35, tzinfo=UTC),
        ),
        _obs(
            spread=25.0,
            stale_age=120.0,
            order_type="moc",
            when=datetime(2026, 4, 1, 20, 30, tzinfo=UTC),
        ),
    ]
    report = compute_calibration_report(
        obs,
        as_of=datetime(2026, 4, 1, tzinfo=UTC),
        floor_bps=1.0,
        min_sample_count=1,
    )
    buckets = {
        (row.spread_bucket, row.stale_price_bucket, row.order_type, row.time_bucket)
        for row in report.buckets
    }
    assert ("tight_spread", "fresh", "limit", "open") in buckets
    assert ("wide_spread", "stale", "moc", "close") in buckets


def test_compute_report_safety_margin_pushes_recommendation() -> None:
    obs = [_obs(slippage=10.0) for _ in range(20)]
    base = compute_calibration_report(
        obs,
        as_of=datetime(2026, 4, 1, tzinfo=UTC),
        floor_bps=1.0,
        min_sample_count=10,
        p90_safety_margin=0.0,
    )
    bumped = compute_calibration_report(
        obs,
        as_of=datetime(2026, 4, 1, tzinfo=UTC),
        floor_bps=1.0,
        min_sample_count=10,
        p90_safety_margin=0.5,
    )
    assert bumped.overall_recommended_bps == pytest.approx(15.0)
    assert bumped.overall_recommended_bps >= base.overall_recommended_bps


def test_calibration_payload_round_trips_json(tmp_path: Path) -> None:
    obs = [_obs(slippage=3.0) for _ in range(5)]
    report = compute_calibration_report(
        obs,
        as_of=datetime(2026, 4, 1, tzinfo=UTC),
        floor_bps=1.0,
        min_sample_count=2,
    )
    output = write_calibration_report(report, tmp_path / "calibration.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 5
    assert payload["overall"]["recommended_bps"] >= 1.0
    assert isinstance(payload["buckets"], list)
    again = calibration_payload(report)
    assert json.dumps(again, sort_keys=True) == json.dumps(payload, sort_keys=True)


def test_compute_report_handles_empty_observations() -> None:
    report = compute_calibration_report(
        [],
        as_of=datetime(2026, 4, 1),
        floor_bps=2.5,
        min_sample_count=5,
    )
    assert isinstance(report, CalibrationReport)
    assert report.sample_count == 0
    assert report.insufficient_data is True
    assert report.overall_recommended_bps == pytest.approx(2.5)
    assert report.buckets == ()


def test_calibration_bucket_dataclass_field_types() -> None:
    bucket = CalibrationBucket(
        tactic="twap",
        adv_bucket="<0.5pct_adv",
        sample_count=10,
        median_slippage_bps=4.0,
        p90_slippage_bps=8.0,
        recommended_bps=8.0,
    )
    assert bucket.tactic == "twap"
    assert bucket.recommended_bps == 8.0
    assert bucket.spread_bucket == "all"
