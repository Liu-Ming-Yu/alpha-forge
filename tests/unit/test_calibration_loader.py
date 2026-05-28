"""Tests for the calibration artifact loader on the CLI side."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.application.research.calibration_artifacts import (
    load_calibration_recommendation_bps,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_payload(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_loader_returns_none_for_missing_path() -> None:
    bps, meta = load_calibration_recommendation_bps(None)
    assert bps is None
    assert meta == {"path": None}


def test_loader_returns_none_for_unreadable_file(tmp_path: Path) -> None:
    target = tmp_path / "broken.json"
    target.write_text("not json", encoding="utf-8")
    bps, meta = load_calibration_recommendation_bps(target)
    assert bps is None
    assert meta["error"].startswith("unreadable")  # type: ignore[index]


def test_loader_rejects_insufficient_data(tmp_path: Path) -> None:
    target = _write_payload(
        tmp_path / "calibration.json",
        {
            "overall": {"recommended_bps": 12.0},
            "sample_count": 4,
            "insufficient_data": True,
            "generated_at": datetime(2026, 4, 1, tzinfo=UTC).isoformat(),
        },
    )
    bps, meta = load_calibration_recommendation_bps(target)
    assert bps is None
    assert meta["error"] == "insufficient_data"


def test_loader_returns_recommendation_when_fresh(tmp_path: Path) -> None:
    as_of = datetime(2026, 4, 10, tzinfo=UTC)
    target = _write_payload(
        tmp_path / "calibration.json",
        {
            "overall": {"recommended_bps": 11.5},
            "sample_count": 200,
            "insufficient_data": False,
            "generated_at": (as_of - timedelta(days=2)).isoformat(),
        },
    )
    bps, meta = load_calibration_recommendation_bps(
        target,
        max_age_days=14.0,
        as_of=as_of,
    )
    assert bps == 11.5
    assert meta["sample_count"] == 200
    assert "error" not in meta


def test_loader_rejects_stale_artifact(tmp_path: Path) -> None:
    as_of = datetime(2026, 5, 1, tzinfo=UTC)
    target = _write_payload(
        tmp_path / "calibration.json",
        {
            "overall": {"recommended_bps": 11.5},
            "sample_count": 200,
            "insufficient_data": False,
            "generated_at": (as_of - timedelta(days=30)).isoformat(),
        },
    )
    bps, meta = load_calibration_recommendation_bps(
        target,
        max_age_days=14.0,
        as_of=as_of,
    )
    assert bps is None
    assert meta["error"] == "stale"
    assert isinstance(meta.get("age_days"), float)


def test_loader_skips_zero_or_negative_recommendation(tmp_path: Path) -> None:
    target = _write_payload(
        tmp_path / "calibration.json",
        {
            "overall": {"recommended_bps": 0.0},
            "sample_count": 100,
            "insufficient_data": False,
            "generated_at": datetime(2026, 4, 1, tzinfo=UTC).isoformat(),
        },
    )
    bps, meta = load_calibration_recommendation_bps(target)
    assert bps is None
    assert meta["error"] == "no overall.recommended_bps"
