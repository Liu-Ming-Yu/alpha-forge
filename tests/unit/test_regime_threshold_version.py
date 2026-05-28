"""Tests for ``MarketRegimeDetector`` threshold versioning."""

from __future__ import annotations

from datetime import UTC, datetime

from quant_platform.services.signal_service.regime_detector import (
    MarketRegimeDetector,
    MarketStats,
    RegimeThresholds,
)


def test_detector_version_starts_with_base() -> None:
    detector = MarketRegimeDetector()
    assert detector.detector_version.startswith(MarketRegimeDetector._BASE_VERSION + "-")


def test_default_thresholds_produce_stable_hash() -> None:
    """Re-instantiating with the same thresholds must produce an identical
    detector_version — the hash is deterministic."""
    d1 = MarketRegimeDetector()
    d2 = MarketRegimeDetector()
    assert d1.detector_version == d2.detector_version


def test_custom_thresholds_produce_distinct_hash() -> None:
    baseline = MarketRegimeDetector()
    tuned = MarketRegimeDetector(
        thresholds=RegimeThresholds(
            crisis_vol=0.40,  # != 0.35 default
        ),
    )
    assert baseline.detector_version != tuned.detector_version


def test_classify_emits_configured_thresholds_in_supporting_features() -> None:
    detector = MarketRegimeDetector(
        thresholds=RegimeThresholds(
            crisis_vol=0.40,
            risk_off_vol=0.28,
            low_vol=0.18,
        ),
    )
    stats = MarketStats(
        trend_z=0.05,
        realized_vol=0.10,
        breadth=0.70,
        as_of=datetime(2026, 4, 23, tzinfo=UTC),
    )
    state = detector.classify(stats)
    assert state.supporting_features["thresholds"]["crisis_vol"] == 0.40
    assert state.supporting_features["thresholds"]["risk_off_vol"] == 0.28
    assert state.detector_version == detector.detector_version


def test_detect_without_update_returns_configured_version() -> None:
    detector = MarketRegimeDetector()
    import asyncio

    state = asyncio.run(detector.detect(datetime(2026, 4, 23, tzinfo=UTC)))
    assert state.detector_version == detector.detector_version
