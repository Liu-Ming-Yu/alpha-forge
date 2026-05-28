"""Unit tests for cross-sectional normalization and feature pipeline."""

from __future__ import annotations

import math
import uuid

import pytest

from quant_platform.services.research_service.features.cross_section.cross_section import (
    STANDARD_FACTOR_SPECS,
    FactorSpec,
    FeatureBundle,
    blend_factors,
    build_feature_bundle,
    rank_normalize,
    winsorize,
    z_score_normalize,
)
from quant_platform.services.research_service.features.factors import InsufficientDataError

# ---------------------------------------------------------------------------
# rank_normalize
# ---------------------------------------------------------------------------


class TestRankNormalize:
    def test_empty_returns_empty(self) -> None:
        assert rank_normalize({}) == {}

    def test_single_element_returns_zero(self) -> None:
        result = rank_normalize({"a": 99.0})
        assert result == {"a": 0.0}

    def test_two_elements(self) -> None:
        result = rank_normalize({"low": 1.0, "high": 2.0})
        assert result["low"] == pytest.approx(-1.0)
        assert result["high"] == pytest.approx(1.0)

    def test_three_elements(self) -> None:
        result = rank_normalize({"a": 1.0, "b": 2.0, "c": 3.0})
        assert result["a"] == pytest.approx(-1.0)
        assert result["b"] == pytest.approx(0.0)
        assert result["c"] == pytest.approx(1.0)

    def test_values_bounded(self) -> None:
        import random

        rng = random.Random(0)
        raw = {i: rng.random() * 1000 for i in range(50)}
        result = rank_normalize(raw)
        assert all(-1.0 <= v <= 1.0 for v in result.values())

    def test_ties_receive_average_rank(self) -> None:
        # Three elements all with the same value → all get rank = 1 (middle of 0,1,2)
        # normalised: 2 * 1 / (3-1) - 1 = 0.0
        result = rank_normalize({"a": 5.0, "b": 5.0, "c": 5.0})
        assert all(v == pytest.approx(0.0) for v in result.values())

    def test_preserves_order(self) -> None:
        vals = {"z": 10.0, "a": 5.0, "m": 7.5}
        result = rank_normalize(vals)
        assert result["a"] < result["m"] < result["z"]

    def test_keys_preserved(self) -> None:
        raw = {uuid.uuid4(): float(i) for i in range(5)}
        result = rank_normalize(raw)
        assert set(result.keys()) == set(raw.keys())


# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------


class TestWinsorize:
    def test_empty_returns_empty(self) -> None:
        assert winsorize({}) == {}

    def test_no_clipping_at_zero(self) -> None:
        raw = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert winsorize(raw, 0.0, 0.0) == raw

    def test_top_outlier_clipped(self) -> None:
        raw = {i: float(i) for i in range(100)}
        result = winsorize(raw, lower_pct=0.0, upper_pct=0.05)
        assert max(result.values()) < max(raw.values())

    def test_bottom_outlier_clipped(self) -> None:
        raw = {i: float(i) for i in range(100)}
        result = winsorize(raw, lower_pct=0.05, upper_pct=0.0)
        assert min(result.values()) > min(raw.values())

    def test_values_not_expand(self) -> None:
        raw = {"a": 1.0, "b": 5.0, "c": 10.0}
        result = winsorize(raw, 0.1, 0.1)
        assert all(v >= min(raw.values()) for v in result.values())
        assert all(v <= max(raw.values()) for v in result.values())


# ---------------------------------------------------------------------------
# z_score_normalize
# ---------------------------------------------------------------------------


class TestZScoreNormalize:
    def test_empty_returns_empty(self) -> None:
        assert z_score_normalize({}) == {}

    def test_constant_returns_zeros(self) -> None:
        result = z_score_normalize({"a": 5.0, "b": 5.0, "c": 5.0})
        assert all(v == pytest.approx(0.0) for v in result.values())

    def test_mean_near_zero(self) -> None:
        raw = {i: float(i) for i in range(100)}
        result = z_score_normalize(raw)
        mean = sum(result.values()) / len(result)
        assert mean == pytest.approx(0.0, abs=1e-9)

    def test_std_near_one(self) -> None:
        raw = {i: float(i) for i in range(100)}
        result = z_score_normalize(raw)
        vals = list(result.values())
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        assert std == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# blend_factors
# ---------------------------------------------------------------------------


class TestBlendFactors:
    def test_empty_returns_empty(self) -> None:
        assert blend_factors([]) == {}

    def test_single_factor_passthrough(self) -> None:
        f = {"a": 0.5, "b": -0.3}
        result = blend_factors([f])
        assert result["a"] == pytest.approx(0.5)
        assert result["b"] == pytest.approx(-0.3)

    def test_equal_weight_average(self) -> None:
        f1 = {"a": 1.0, "b": 0.0}
        f2 = {"a": 0.0, "b": 1.0}
        result = blend_factors([f1, f2])
        assert result["a"] == pytest.approx(0.5)
        assert result["b"] == pytest.approx(0.5)

    def test_custom_weights(self) -> None:
        f1 = {"a": 1.0}
        f2 = {"a": 0.0}
        result = blend_factors([f1, f2], weights=[3.0, 1.0])
        assert result["a"] == pytest.approx(0.75)

    def test_missing_key_treated_as_neutral(self) -> None:
        f1 = {"a": 1.0}
        f2 = {"b": 1.0}
        result = blend_factors([f1, f2])
        # "a" appears in f1 but not f2 → 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert result["a"] == pytest.approx(0.5)
        assert result["b"] == pytest.approx(0.5)

    def test_zero_weights_returns_zero_scores(self) -> None:
        result = blend_factors([{"a": 1.0}], weights=[0.0])
        assert result["a"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FactorSpec and STANDARD_FACTOR_SPECS
# ---------------------------------------------------------------------------


def test_standard_factor_specs_names() -> None:
    names = {spec.name for spec in STANDARD_FACTOR_SPECS}
    assert "momentum_1m" in names
    assert "momentum_3m" in names
    assert "momentum_12m_1m" in names
    assert "vol_compression" in names
    assert "realized_vol_21d" in names
    # Non-momentum diversification factors registered by the Parity sprint.
    assert "short_term_reversal_5d" in names
    assert "trend_quality_63d" in names
    assert "distance_to_52w_high" in names


def test_non_momentum_factors_are_alpha() -> None:
    alpha_specs = {s.name for s in STANDARD_FACTOR_SPECS if s.is_alpha}
    assert "short_term_reversal_5d" in alpha_specs
    assert "trend_quality_63d" in alpha_specs
    assert "distance_to_52w_high" in alpha_specs


def test_realized_vol_is_not_alpha() -> None:
    vol_spec = next(s for s in STANDARD_FACTOR_SPECS if s.name == "realized_vol_21d")
    assert vol_spec.is_alpha is False


def test_momentum_specs_are_alpha() -> None:
    alpha_specs = [s for s in STANDARD_FACTOR_SPECS if s.is_alpha]
    names = {s.name for s in alpha_specs}
    assert "momentum_1m" in names
    assert "momentum_3m" in names
    assert "momentum_12m_1m" in names


def test_factor_spec_compute_raises_insufficient_data() -> None:
    spec = next(s for s in STANDARD_FACTOR_SPECS if s.name == "momentum_12m_1m")
    with pytest.raises(InsufficientDataError):
        spec.compute([100.0] * 5)


# ---------------------------------------------------------------------------
# build_feature_bundle
# ---------------------------------------------------------------------------


def _make_bar_data(
    n_instruments: int,
    n_bars: int,
    seed: int = 0,
) -> dict[uuid.UUID, list[float]]:
    """Build synthetic bar data with random-walk prices."""
    import random

    rng = random.Random(seed)
    data = {}
    for _ in range(n_instruments):
        instr_id = uuid.uuid4()
        price = 100.0
        closes = [price]
        for _ in range(n_bars - 1):
            price *= 1 + rng.gauss(0.0005, 0.01)
            closes.append(max(price, 1.0))
        data[instr_id] = closes
    return data


class TestBuildFeatureBundle:
    def test_returns_feature_bundle(self) -> None:
        bar_data = _make_bar_data(20, 300)
        bundle = build_feature_bundle(bar_data)
        assert isinstance(bundle, FeatureBundle)

    def test_alpha_features_present(self) -> None:
        bar_data = _make_bar_data(20, 300)
        bundle = build_feature_bundle(bar_data)
        assert len(bundle.alpha_features) > 0

    def test_alpha_feature_keys_match_alpha_specs(self) -> None:
        bar_data = _make_bar_data(5, 300)
        bundle = build_feature_bundle(bar_data)
        expected_keys = {s.name for s in STANDARD_FACTOR_SPECS if s.is_alpha}
        for feat_dict in bundle.alpha_features.values():
            assert set(feat_dict.keys()) == expected_keys

    def test_alpha_features_in_range(self) -> None:
        bar_data = _make_bar_data(30, 300)
        bundle = build_feature_bundle(bar_data)
        for feat_dict in bundle.alpha_features.values():
            for v in feat_dict.values():
                assert -1.0 <= v <= 1.0, f"feature value {v} out of range"

    def test_vol_forecasts_positive(self) -> None:
        bar_data = _make_bar_data(10, 300)
        bundle = build_feature_bundle(bar_data)
        assert len(bundle.vol_forecasts) > 0
        for v in bundle.vol_forecasts.values():
            assert v > 0.0

    def test_instruments_with_insufficient_data_excluded(self) -> None:
        bar_data = _make_bar_data(10, 300)
        # Add one instrument with only 10 bars (not enough for any momentum factor)
        short_id = uuid.uuid4()
        bar_data[short_id] = [100.0] * 10
        bundle = build_feature_bundle(bar_data)
        # The short instrument should not appear in alpha_features
        assert short_id not in bundle.alpha_features

    def test_partial_factor_coverage_omits_missing_factor(self) -> None:
        """An instrument retained by majority rule but missing one alpha
        factor must have that factor *omitted* from its feature dict —
        not zero-filled — so downstream scoring decrements coverage and
        confidence rather than treating the missing factor as neutral.
        """
        # 30-bar history is enough for short_term_reversal_5d (needs 6 bars)
        # and realized_vol_21d (needs 22) but NOT for momentum_1m (22) is
        # actually OK, momentum_3m (~63) and momentum_6m (~126) are NOT.
        bar_data = _make_bar_data(5, 200)
        partial_id = uuid.uuid4()
        bar_data[partial_id] = [100.0 + i * 0.1 for i in range(40)]  # 40 bars
        bundle = build_feature_bundle(bar_data)

        if partial_id not in bundle.alpha_features:
            # Acceptable outcome: dropped entirely; this test only asserts
            # that *if* retained, missing factors are omitted (not zero).
            return

        feat = bundle.alpha_features[partial_id]
        all_alpha = {s.name for s in STANDARD_FACTOR_SPECS if s.is_alpha}
        missing = all_alpha - set(feat)
        assert missing, "test setup expected at least one factor to be missing"
        # Omitted, not zero-filled: the key must be absent.
        for name in missing:
            assert name not in feat

    def test_instrument_with_full_history_is_included(self) -> None:
        """Upper-bound guard: an instrument with enough bars for every factor
        must not be dropped by the majority-hit rule.  Pairs with the
        ``insufficient_data`` test above so a future tightening cannot
        silently drop healthy instruments as a collateral effect."""
        bar_data = _make_bar_data(3, 300)
        full_id = uuid.uuid4()
        bar_data[full_id] = [100.0 + i * 0.1 for i in range(300)]
        bundle = build_feature_bundle(bar_data)
        assert full_id in bundle.alpha_features

    def test_empty_bar_data_returns_empty_bundle(self) -> None:
        bundle = build_feature_bundle({})
        assert bundle.alpha_features == {}
        assert bundle.vol_forecasts == {}

    def test_custom_factor_spec(self) -> None:
        from quant_platform.services.research_service.features.factors import momentum_return

        specs = [
            FactorSpec(
                name="mom_5d",
                compute=lambda c: momentum_return(c, 5),
                is_alpha=True,
            )
        ]
        bar_data = _make_bar_data(10, 30)
        bundle = build_feature_bundle(bar_data, factor_specs=specs)
        for feat_dict in bundle.alpha_features.values():
            assert "mom_5d" in feat_dict

    def test_realized_vol_not_in_alpha_features(self) -> None:
        bar_data = _make_bar_data(10, 300)
        bundle = build_feature_bundle(bar_data)
        for feat_dict in bundle.alpha_features.values():
            assert "realized_vol_21d" not in feat_dict
