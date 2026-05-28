from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features import FeatureFrame, FeatureRegistry, FeatureSpec
from quant_platform.research.features.price_volume import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    PriceVolumeConfig,
    compute_price_volume_features,
)
from quant_platform.research.features.transforms import safe_div


@pytest.fixture
def single_instrument_bars() -> pd.DataFrame:
    rows = []
    for idx, date in enumerate(pd.bdate_range("2024-01-02", periods=300)):
        close = 100.0 + idx
        rows.append(
            {
                "instrument_id": "AAA",
                "date": date,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000 + idx,
            }
        )
    return pd.DataFrame(rows)


def _spec(name: str, version: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family="price_volume",
        description="test feature",
        expected_direction="unknown",
        required_inputs=("close",),
        point_in_time=True,
        lookback_days=1,
        version=version,
        larger_is_better=False,
    )


def test_custom_config_version_flows_into_feature_specs(
    single_instrument_bars: pd.DataFrame,
) -> None:
    cfg = PriceVolumeConfig(version="price-volume-test-v0")
    result = compute_price_volume_features(single_instrument_bars, config=cfg)

    assert all(spec.version == "price-volume-test-v0" for spec in result.feature_specs.values())


def test_registry_allows_same_feature_name_across_versions() -> None:
    registry = FeatureRegistry()
    v1 = _spec("ret_21d", "price-volume-starter-v1")
    v2 = _spec("ret_21d", "price-volume-experimental-v2")

    registry.register(v1)
    registry.register(v2)

    assert registry.get("ret_21d", "price-volume-starter-v1") == v1
    assert registry.get("ret_21d", "price-volume-experimental-v2") == v2
    assert registry.get_latest("ret_21d") == v2
    assert registry.by_version("price-volume-starter-v1") == (v1,)
    assert registry.names() == ("ret_21d",)


def test_safe_div_default_masks_zero_nan_negative_and_tiny_denominators() -> None:
    numer = pd.Series([10.0, 10.0, 10.0, 10.0])
    denom = pd.Series([0.0, np.nan, -2.0, 1e-12])

    out = safe_div(numer, denom, min_abs_denom=1e-6)

    assert out.isna().all()


def test_safe_div_can_deliberately_allow_negative_denominators() -> None:
    numer = pd.Series([10.0])
    denom = pd.Series([-2.0])

    out = safe_div(numer, denom, require_positive_denom=False)

    assert out.iloc[0] == pytest.approx(-5.0)


def test_feature_frame_validates_columns_specs_and_coverage() -> None:
    spec = _spec("ret_21d", "price-volume-starter-v1")

    with pytest.raises(ValueError, match="duplicate feature_names"):
        FeatureFrame(
            frame=pd.DataFrame(
                {"instrument_id": ["AAA"], "date": [pd.Timestamp("2024-01-02")], "ret_21d": [1.0]}
            ),
            feature_names=("ret_21d", "ret_21d"),
            feature_specs={"ret_21d": spec},
            coverage={"ret_21d": 1},
            key_columns=("instrument_id", "date"),
        )

    with pytest.raises(ValueError, match="missing feature columns"):
        FeatureFrame(
            frame=pd.DataFrame({"instrument_id": ["AAA"], "date": [pd.Timestamp("2024-01-02")]}),
            feature_names=("ret_21d",),
            feature_specs={"ret_21d": spec},
            coverage={"ret_21d": 1},
            key_columns=("instrument_id", "date"),
        )

    with pytest.raises(ValueError, match="coverage keys"):
        FeatureFrame(
            frame=pd.DataFrame(
                {"instrument_id": ["AAA"], "date": [pd.Timestamp("2024-01-02")], "ret_21d": [1.0]}
            ),
            feature_names=("ret_21d",),
            feature_specs={"ret_21d": spec},
            coverage={},
            key_columns=("instrument_id", "date"),
        )


def test_duplicate_high_watermark_alias_is_excluded_from_default_training_features(
    single_instrument_bars: pd.DataFrame,
) -> None:
    result = compute_price_volume_features(single_instrument_bars)

    assert "distance_to_52w_high" in FEATURE_NAMES
    assert "drawdown_from_252d_high" in FEATURE_NAMES
    assert "distance_to_52w_high" in result.training_feature_names
    assert "drawdown_from_252d_high" not in result.training_feature_names
    assert "drawdown_from_252d_high" not in DEFAULT_TRAINING_FEATURE_NAMES
    assert result.feature_specs["drawdown_from_252d_high"].canonical_name == "distance_to_52w_high"


def test_price_volume_specs_declare_eod_after_close_signal_timestamp(
    single_instrument_bars: pd.DataFrame,
) -> None:
    result = compute_price_volume_features(single_instrument_bars)

    assert all(spec.signal_timestamp == "eod_after_close" for spec in result.feature_specs.values())
