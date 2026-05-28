"""Unit tests for classical alpha factor functions."""

from __future__ import annotations

import pytest

from quant_platform.services.research_service.features.factors import (
    InsufficientDataError,
    distance_to_52w_high,
    momentum_return,
    momentum_skip1m,
    realized_vol,
    short_term_reversal,
    sma,
    trend_quality,
    trend_z_score,
    vol_compression_ratio,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _geometric_series(start: float, daily_return: float, n: int) -> list[float]:
    """Build a price series with a constant daily return."""
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + daily_return))
    return closes


# ---------------------------------------------------------------------------
# momentum_return
# ---------------------------------------------------------------------------


class TestMomentumReturn:
    def test_flat_series_returns_zero(self) -> None:
        closes = [100.0] * 22
        assert momentum_return(closes, 21) == pytest.approx(0.0)

    def test_known_return(self) -> None:
        closes = [100.0, 110.0]
        assert momentum_return(closes, 1) == pytest.approx(0.10)

    def test_negative_return(self) -> None:
        closes = [100.0, 90.0]
        assert momentum_return(closes, 1) == pytest.approx(-0.10)

    def test_period_21_uses_correct_base(self) -> None:
        # 22 bars: base at index 0 (100), end at index 21 (150)
        closes = [100.0] + [0.0] * 20 + [150.0]
        assert momentum_return(closes, 21) == pytest.approx(0.50)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            momentum_return([100.0] * 5, period=10)

    def test_negative_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period must be positive"):
            momentum_return([100.0, 110.0], period=0)

    def test_geometric_series_compound_return(self) -> None:
        # 1% daily for 21 days
        closes = _geometric_series(100.0, 0.01, 22)
        expected = 100.0 * (1.01**21) / 100.0 - 1.0
        assert momentum_return(closes, 21) == pytest.approx(expected, rel=1e-6)

    def test_zero_end_price_raises_insufficient_data(self) -> None:
        # Delisting: last close drops to 0.0; a -100% return would propagate
        # an "extreme momentum" signal. Treat as insufficient data instead.
        closes = [100.0] * 21 + [0.0]
        with pytest.raises(InsufficientDataError, match="end price is zero"):
            momentum_return(closes, period=21)


# ---------------------------------------------------------------------------
# momentum_skip1m
# ---------------------------------------------------------------------------


class TestMomentumSkip1m:
    def test_flat_returns_zero(self) -> None:
        closes = [100.0] * 253
        assert momentum_skip1m(closes) == pytest.approx(0.0)

    def test_known_skip_momentum(self) -> None:
        # 253 bars: index 0 = 100, index 231 (-(21+1)) = 150, rest don't matter
        closes = [100.0] * 253
        closes[252 - 252] = 100.0  # start: index -(252+1) = index 0
        closes[252 - 21] = 150.0  # end: index -(21+1)
        # momentum_skip1m returns closes[-(skip+1)] / closes[-(long+1)] - 1
        expected = closes[252 - 21] / closes[252 - 252] - 1.0
        assert momentum_skip1m(closes, 252, 21) == pytest.approx(expected)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            momentum_skip1m([100.0] * 50, long_period=252, skip_period=21)

    def test_long_lte_skip_raises(self) -> None:
        with pytest.raises(ValueError, match="long_period"):
            momentum_skip1m([100.0] * 300, long_period=21, skip_period=21)

    def test_skip_avoids_reversal(self) -> None:
        # Price trends up for 252 days then drops in the last 21 days
        closes = _geometric_series(100.0, 0.002, 232)  # 232 bars of +0.2%/day
        # Append 21 bars of strong reversal
        for _ in range(21):
            closes.append(closes[-1] * 0.99)  # -1%/day
        assert len(closes) == 253
        skip_mom = momentum_skip1m(closes, 252, 21)
        raw_mom = momentum_return(closes, 252)
        # skip-1m should be more positive (misses the reversal period)
        assert skip_mom > raw_mom


# ---------------------------------------------------------------------------
# realized_vol
# ---------------------------------------------------------------------------


class TestRealizedVol:
    def test_constant_prices_returns_zero_vol(self) -> None:
        closes = [100.0] * 23
        assert realized_vol(closes, window=21) == pytest.approx(0.0)

    def test_annualised_output_scale(self) -> None:
        # 1% daily vol → annualised ≈ 1% × sqrt(252)
        daily_vol_pct = 0.01
        closes = [100.0]
        import random

        rng = random.Random(42)
        for _ in range(252):
            closes.append(closes[-1] * (1 + rng.gauss(0, daily_vol_pct)))
        ann_vol = realized_vol(closes, window=21, annualize=True)
        # Allow wide tolerance for random data
        assert 0.01 < ann_vol < 0.50

    def test_non_annualised_less_than_annualised(self) -> None:
        closes = _geometric_series(100.0, 0.001, 30)
        closes[-1] *= 1.05  # add a jump to get nonzero vol
        non_ann = realized_vol(closes, window=21, annualize=False)
        ann = realized_vol(closes, window=21, annualize=True)
        assert ann > non_ann

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            realized_vol([100.0] * 5, window=21)

    def test_minimum_required_bars(self) -> None:
        closes = [100.0 + i for i in range(22)]  # exactly 22 bars for window=21
        vol = realized_vol(closes, window=21)
        assert vol >= 0.0

    def test_known_vol_two_bars(self) -> None:
        # Two bars: one log return
        # log(110/100) = log(1.1) ≈ 0.09531
        closes = [100.0, 110.0]
        vol_daily = realized_vol([100.0, 110.0], window=1, annualize=False)
        assert vol_daily == pytest.approx(0.0, abs=1e-9)  # single return → std=0


# ---------------------------------------------------------------------------
# vol_compression_ratio
# ---------------------------------------------------------------------------


class TestVolCompressionRatio:
    def test_constant_prices_returns_one(self) -> None:
        closes = [100.0] * 30
        assert vol_compression_ratio(closes) == pytest.approx(1.0)

    def test_compressed_vol_less_than_one(self) -> None:
        # Price trending up smoothly for 21 days; last 5 days are flat
        closes = _geometric_series(100.0, 0.005, 22)
        # Flatten the last 5 days
        for _ in range(5):
            closes.append(closes[-1])
        result = vol_compression_ratio(closes, short_window=5, long_window=21)
        assert result < 1.0

    def test_expanding_vol_greater_than_one(self) -> None:
        # Flat for most of the series, then large moves at the end
        closes = [100.0] * 22
        closes[-5] = 105.0
        closes[-4] = 95.0
        closes[-3] = 108.0
        closes[-2] = 92.0
        closes[-1] = 107.0
        result = vol_compression_ratio(closes, short_window=5, long_window=21)
        assert result > 1.0

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            vol_compression_ratio([100.0] * 10, short_window=5, long_window=21)

    def test_long_lte_short_raises(self) -> None:
        with pytest.raises(ValueError, match="long_window"):
            vol_compression_ratio([100.0] * 30, short_window=21, long_window=5)


# ---------------------------------------------------------------------------
# sma
# ---------------------------------------------------------------------------


class TestSMA:
    def test_arithmetic_mean(self) -> None:
        closes = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert sma(closes, window=5) == pytest.approx(3.0)

    def test_uses_last_window_bars(self) -> None:
        closes = [999.0, 1.0, 2.0, 3.0]  # first bar is an outlier
        assert sma(closes, window=3) == pytest.approx(2.0)

    def test_window_one(self) -> None:
        closes = [42.0, 100.0, 200.0]
        assert sma(closes, window=1) == pytest.approx(200.0)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            sma([1.0, 2.0], window=5)

    def test_non_positive_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be positive"):
            sma([1.0, 2.0, 3.0], window=0)


# ---------------------------------------------------------------------------
# trend_z_score
# ---------------------------------------------------------------------------


class TestTrendZScore:
    def test_price_at_sma_returns_zero(self) -> None:
        closes = [100.0] * 200
        assert trend_z_score(closes, window=200) == pytest.approx(0.0)

    def test_price_above_sma_positive(self) -> None:
        # 199 bars at 100, then one bar at 200 (well above 200-day MA of ~100)
        closes = [100.0] * 199 + [200.0]
        z = trend_z_score(closes, window=200)
        assert z > 0.0

    def test_price_below_sma_negative(self) -> None:
        closes = [100.0] * 199 + [50.0]
        z = trend_z_score(closes, window=200)
        assert z < 0.0

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            trend_z_score([100.0] * 50, window=200)

    def test_known_value(self) -> None:
        # SMA(5) = (1+2+3+4+5)/5 = 3.0; current close = 5.0
        # trend_z = (5 - 3) / 3 ≈ 0.6667
        closes = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected = (5.0 - 3.0) / 3.0
        assert trend_z_score(closes, window=5) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# short_term_reversal
# ---------------------------------------------------------------------------


class TestShortTermReversal:
    def test_is_negated_momentum_return(self) -> None:
        closes = _geometric_series(start=100.0, daily_return=0.01, n=10)
        assert short_term_reversal(closes, window=5) == pytest.approx(
            -momentum_return(closes, period=5)
        )

    def test_recent_downturn_yields_positive_score(self) -> None:
        # Name sold off the last 5 days — reversal factor should be positive.
        closes = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0]
        score = short_term_reversal(closes, window=5)
        assert score > 0.0
        assert score == pytest.approx(0.05)

    def test_flat_series_returns_zero(self) -> None:
        closes = [100.0] * 6
        assert short_term_reversal(closes, window=5) == pytest.approx(0.0)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            short_term_reversal([100.0, 101.0, 102.0], window=5)


# ---------------------------------------------------------------------------
# trend_quality
# ---------------------------------------------------------------------------


class TestTrendQuality:
    def test_constant_positive_return_scores_high(self) -> None:
        # A perfectly smooth uptrend: std ~ 0, mean > 0 -> large ratio.
        closes = _geometric_series(start=100.0, daily_return=0.01, n=100)
        score = trend_quality(closes, window=63)
        assert score > 100.0  # huge when vol ~ 0

    def test_flat_series_returns_zero(self) -> None:
        closes = [100.0] * 100
        assert trend_quality(closes, window=63) == pytest.approx(0.0)

    def test_symmetric_oscillation_near_zero(self) -> None:
        # Alternating +-1% returns: mean ≈ 0, std > 0 -> score near 0.
        closes = [100.0]
        for i in range(100):
            ret = 0.01 if i % 2 == 0 else -0.01
            closes.append(closes[-1] * (1 + ret))
        score = trend_quality(closes, window=63)
        assert abs(score) < 0.5

    def test_downtrend_yields_negative(self) -> None:
        closes = _geometric_series(start=100.0, daily_return=-0.01, n=100)
        assert trend_quality(closes, window=63) < 0.0

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            trend_quality([100.0] * 10, window=63)

    def test_non_positive_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be positive"):
            trend_quality([100.0] * 10, window=0)


# ---------------------------------------------------------------------------
# distance_to_52w_high
# ---------------------------------------------------------------------------


class TestDistanceTo52wHigh:
    def test_new_high_returns_zero(self) -> None:
        closes = [100.0 + i for i in range(252)]  # monotonic uptrend ending at peak
        assert distance_to_52w_high(closes, window=252) == pytest.approx(0.0)

    def test_below_peak_returns_negative(self) -> None:
        closes = [50.0] * 100 + [200.0] + [150.0] * 151  # peak at mid-window
        dist = distance_to_52w_high(closes, window=252)
        assert dist == pytest.approx((150.0 - 200.0) / 200.0)

    def test_always_non_positive(self) -> None:
        closes = [100.0, 200.0, 150.0, 175.0, 90.0]
        assert distance_to_52w_high(closes, window=5) <= 0.0

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            distance_to_52w_high([100.0] * 5, window=252)

    def test_non_positive_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be positive"):
            distance_to_52w_high([100.0, 200.0], window=0)
