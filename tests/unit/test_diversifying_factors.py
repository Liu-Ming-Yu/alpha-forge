"""WS4 diversifying factors: low-volatility and mean-reversion."""

from __future__ import annotations

import pytest

from quant_platform.core.algorithms.price_factors import low_volatility, mean_reversion
from quant_platform.services.research_service.features.cross_section.cross_section_factors import (
    STANDARD_FACTOR_SPECS,
)

_NEW_FACTORS = {"reversal_21d", "low_volatility_63d", "mean_reversion_63d"}


def test_low_volatility_higher_for_calmer_series() -> None:
    calm = [100.0 + 0.01 * i for i in range(120)]
    choppy = [100.0 + (5.0 if i % 2 else -5.0) for i in range(120)]
    # Calmer name scores higher (less negative) — the low-vol anomaly.
    assert low_volatility(calm, 63) > low_volatility(choppy, 63)


def test_low_volatility_flat_series_is_zero() -> None:
    assert low_volatility([100.0] * 120, 63) == pytest.approx(0.0)


def test_mean_reversion_positive_below_sma() -> None:
    closes = [110.0] * 62 + [100.0]  # last well below the 63-day average
    assert mean_reversion(closes, 63) > 0.0


def test_mean_reversion_negative_above_sma() -> None:
    closes = [100.0] * 62 + [120.0]  # last well above the 63-day average
    assert mean_reversion(closes, 63) < 0.0


def test_diversifying_factor_specs_registered_as_alpha() -> None:
    by_name = {spec.name: spec for spec in STANDARD_FACTOR_SPECS}
    assert set(by_name) >= _NEW_FACTORS
    for name in _NEW_FACTORS:
        assert by_name[name].is_alpha


def test_new_factor_specs_compute_on_a_price_series() -> None:
    by_name = {spec.name: spec for spec in STANDARD_FACTOR_SPECS}
    closes = [100.0 + 0.05 * i for i in range(300)]
    for name in _NEW_FACTORS:
        value = by_name[name].compute(closes)
        assert isinstance(value, float)
