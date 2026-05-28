"""Unit tests for MarketRegimeDetector and MarketStats."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from quant_platform.core.domain.signals import RegimeLabel
from quant_platform.services.signal_service.regime_detector import (
    MarketRegimeDetector,
    MarketStats,
    RegimeThresholds,
    _classify,
)

_UTC = UTC
_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=_UTC)


def _stats(trend_z: float, vol: float, breadth: float) -> MarketStats:
    return MarketStats(
        trend_z=trend_z,
        realized_vol=vol,
        breadth=breadth,
        as_of=_NOW,
    )


# ---------------------------------------------------------------------------
# MarketStats validation
# ---------------------------------------------------------------------------


class TestMarketStats:
    def test_valid_stats(self) -> None:
        s = _stats(0.05, 0.15, 0.60)
        assert s.trend_z == 0.05

    def test_breadth_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="breadth"):
            _stats(0.0, 0.15, -0.1)

    def test_breadth_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="breadth"):
            _stats(0.0, 0.15, 1.5)

    def test_negative_vol_raises(self) -> None:
        with pytest.raises(ValueError, match="realized_vol"):
            _stats(0.0, -0.01, 0.5)

    def test_naive_datetime_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            MarketStats(
                trend_z=0.0,
                realized_vol=0.15,
                breadth=0.5,
                as_of=datetime(2026, 1, 5, 14, 0),  # naive
            )


# ---------------------------------------------------------------------------
# Regime classification rules
# ---------------------------------------------------------------------------


_T = RegimeThresholds()  # default thresholds


class TestClassifyRules:
    def test_extreme_vol_is_crisis(self) -> None:
        label, _ = _classify(_stats(0.10, 0.40, 0.70), _T)
        assert label == RegimeLabel.CRISIS

    def test_high_vol_and_downtrend_is_crisis(self) -> None:
        # vol = 0.30 (≥ 0.25) AND trend_z = -0.10 (≤ -0.05)
        label, _ = _classify(_stats(-0.10, 0.30, 0.50), _T)
        assert label == RegimeLabel.CRISIS

    def test_high_vol_alone_is_risk_off(self) -> None:
        # vol ≥ 0.25 but trend is neutral and breadth is OK
        label, _ = _classify(_stats(0.05, 0.27, 0.60), _T)
        assert label == RegimeLabel.RISK_OFF

    def test_downtrend_alone_is_risk_off(self) -> None:
        # vol is low but market is below MA
        label, _ = _classify(_stats(-0.08, 0.14, 0.55), _T)
        assert label == RegimeLabel.RISK_OFF

    def test_weak_breadth_alone_is_risk_off(self) -> None:
        label, _ = _classify(_stats(0.05, 0.18, 0.35), _T)
        assert label == RegimeLabel.RISK_OFF

    def test_all_green_is_risk_on(self) -> None:
        # vol < 0.20, trend > 0.02, breadth > 0.55
        label, conf = _classify(_stats(0.08, 0.14, 0.65), _T)
        assert label == RegimeLabel.RISK_ON
        assert conf > 0.6

    def test_borderline_is_transition(self) -> None:
        # vol between low_vol (0.20) and risk_off_vol (0.25), trend slightly positive
        label, _ = _classify(_stats(0.03, 0.22, 0.52), _T)
        assert label == RegimeLabel.TRANSITION

    def test_crisis_confidence_high(self) -> None:
        _, conf = _classify(_stats(0.0, 0.50, 0.30), _T)
        assert conf >= 0.80

    def test_risk_on_confidence_above_0_6(self) -> None:
        _, conf = _classify(_stats(0.10, 0.12, 0.70), _T)
        assert conf >= 0.60

    def test_confidence_bounded_to_one(self) -> None:
        for trend in (-0.5, 0.0, 0.5):
            for vol in (0.05, 0.20, 0.40):
                for breadth in (0.2, 0.5, 0.8):
                    _, conf = _classify(_stats(trend, vol, breadth), _T)
                    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------


def test_custom_thresholds_change_classification() -> None:
    tight = RegimeThresholds(risk_off_vol=0.10)  # very tight
    loose = RegimeThresholds(risk_off_vol=0.40)  # very loose

    stats = _stats(0.05, 0.15, 0.60)
    label_tight, _ = _classify(stats, tight)
    label_loose, _ = _classify(stats, loose)

    # With tight threshold, 15% vol triggers RISK_OFF
    assert label_tight == RegimeLabel.RISK_OFF
    # With loose threshold, 15% vol is fine
    assert label_loose in (RegimeLabel.RISK_ON, RegimeLabel.TRANSITION)


# ---------------------------------------------------------------------------
# MarketRegimeDetector class
# ---------------------------------------------------------------------------


class TestMarketRegimeDetector:
    def test_classify_returns_regime_state(self) -> None:
        detector = MarketRegimeDetector()
        state = detector.classify(_stats(0.08, 0.14, 0.65))
        assert state.regime_label == RegimeLabel.RISK_ON
        assert 0.0 <= state.confidence <= 1.0
        assert state.detector_version.startswith("v1.0")

    def test_classify_includes_supporting_features(self) -> None:
        detector = MarketRegimeDetector()
        state = detector.classify(_stats(0.08, 0.14, 0.65))
        assert "trend_z" in state.supporting_features
        assert "realized_vol" in state.supporting_features
        assert "breadth" in state.supporting_features

    @pytest.mark.asyncio
    async def test_detect_without_update_returns_transition(self) -> None:
        detector = MarketRegimeDetector()
        state = await detector.detect(_NOW)
        assert state.regime_label == RegimeLabel.TRANSITION
        assert state.confidence == 0.0

    @pytest.mark.asyncio
    async def test_detect_after_update_reflects_stats(self) -> None:
        # stability_window=1 means the regime is declared immediately on the
        # first consistent candidate, allowing single-call tests without
        # having to fill the stability buffer.
        detector = MarketRegimeDetector(RegimeThresholds(stability_window=1))
        # Crisis conditions
        detector.update(_stats(trend_z=-0.10, vol=0.40, breadth=0.25))
        state = await detector.detect(_NOW)
        assert state.regime_label == RegimeLabel.CRISIS

    @pytest.mark.asyncio
    async def test_detect_uses_provided_as_of(self) -> None:
        detector = MarketRegimeDetector()
        detector.update(_stats(0.05, 0.15, 0.60))
        custom_time = datetime(2026, 6, 1, 9, 30, tzinfo=_UTC)
        state = await detector.detect(custom_time)
        assert state.as_of == custom_time

    def test_update_replaces_previous_stats(self) -> None:
        detector = MarketRegimeDetector()
        detector.update(_stats(0.08, 0.14, 0.70))  # RISK_ON
        state1 = detector.classify(detector._current_stats)  # type: ignore[arg-type]
        assert state1.regime_label == RegimeLabel.RISK_ON

        detector.update(_stats(-0.10, 0.40, 0.25))  # CRISIS
        state2 = detector.classify(detector._current_stats)  # type: ignore[arg-type]
        assert state2.regime_label == RegimeLabel.CRISIS

    def test_regime_id_is_unique_per_call(self) -> None:
        detector = MarketRegimeDetector()
        s = _stats(0.05, 0.15, 0.60)
        state1 = detector.classify(s)
        state2 = detector.classify(s)
        assert state1.regime_id != state2.regime_id


# ---------------------------------------------------------------------------
# compute_stats static method
# ---------------------------------------------------------------------------


class TestComputeStats:
    def _make_spy_closes(self, n: int, start: float = 100.0, drift: float = 0.001) -> list[float]:
        import random

        rng = random.Random(1)
        closes = [start]
        for _ in range(n - 1):
            closes.append(closes[-1] * (1 + rng.gauss(drift, 0.008)))
        return closes

    def test_returns_market_stats(self) -> None:
        spy = self._make_spy_closes(220)
        instruments = {uuid.uuid4(): self._make_spy_closes(60) for _ in range(20)}
        stats = MarketRegimeDetector.compute_stats(spy, instruments, _NOW)
        assert isinstance(stats, MarketStats)

    def test_breadth_between_zero_and_one(self) -> None:
        spy = self._make_spy_closes(220)
        instruments = {uuid.uuid4(): self._make_spy_closes(60) for _ in range(50)}
        stats = MarketRegimeDetector.compute_stats(spy, instruments, _NOW)
        assert 0.0 <= stats.breadth <= 1.0

    def test_vol_positive(self) -> None:
        spy = self._make_spy_closes(220)
        stats = MarketRegimeDetector.compute_stats(spy, {}, _NOW)
        assert stats.realized_vol > 0.0

    def test_short_index_series_falls_back_to_neutral(self) -> None:
        # Only 5 bars — both trend and vol should fall back to defaults
        spy = [100.0] * 5
        stats = MarketRegimeDetector.compute_stats(spy, {}, _NOW)
        assert stats.trend_z == 0.0  # fallback
        assert stats.realized_vol == 0.20  # fallback
        assert stats.breadth == 0.5  # no instruments

    def test_all_instruments_above_ma_breadth_one(self) -> None:
        # 50 bars: steadily rising series → all above 50-day MA
        spy = self._make_spy_closes(220)
        rising = [float(i + 100) for i in range(51)]  # strictly rising
        instruments = {uuid.uuid4(): rising for _ in range(10)}
        stats = MarketRegimeDetector.compute_stats(spy, instruments, _NOW, breadth_window=50)
        assert stats.breadth == pytest.approx(1.0)
