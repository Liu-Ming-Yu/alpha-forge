"""Unit tests for the NNLS + L2 factor-weight calibrator."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quant_platform.services.research_service.features.factors.calibration import (
    ALPHA_BLOC,
    MOMENTUM_BLOC,
    CalibratedWeights,
    CalibrationSample,
    calibrate,
    read_artifact,
    write_artifact,
)

_UTC = UTC


def _synth_samples(
    n: int,
    true_weights: dict[str, float],
    noise: float = 0.01,
    seed: int = 0,
) -> list[CalibrationSample]:
    rng = np.random.default_rng(seed)
    samples: list[CalibrationSample] = []
    all_factors = list(MOMENTUM_BLOC) + list(ALPHA_BLOC)
    for i in range(n):
        features = {name: float(rng.uniform(-1, 1)) for name in all_factors}
        signal = sum(features[name] * true_weights.get(name, 0.0) for name in all_factors)
        ret = signal + rng.normal(0, noise)
        samples.append(
            CalibrationSample(
                as_of=datetime(2024, 1, 1, tzinfo=_UTC) + timedelta(days=i),
                instrument_id=uuid.uuid4(),
                features=features,
                forward_return=float(ret),
            )
        )
    return samples


def test_calibrate_recovers_dominant_momentum_factor() -> None:
    true = {
        "momentum_1m": 0.3,
        "momentum_3m": 0.4,
        "momentum_12m_1m": 0.2,
        "vol_compression": 0.1,
        "short_term_reversal_5d": 0.05,
        "trend_quality_63d": 0.03,
        "distance_to_52w_high": 0.02,
    }
    samples = _synth_samples(400, true, noise=0.02, seed=42)
    result = calibrate(samples)

    assert set(result.weights.keys()) == set(true.keys())
    # momentum bloc sums to ~0.9, alpha bloc to ~0.1
    mom_sum = sum(result.weights[k] for k in MOMENTUM_BLOC)
    alpha_sum = sum(result.weights[k] for k in ALPHA_BLOC)
    assert abs(mom_sum - 0.90) < 1e-6
    assert abs(alpha_sum - 0.10) < 1e-6


def test_calibrate_returns_nonnegative_weights() -> None:
    # Flip the sign on one factor so NNLS has to clip it to 0.
    true = {
        "momentum_1m": 0.4,
        "momentum_3m": 0.3,
        "momentum_12m_1m": 0.2,
        "vol_compression": -0.5,  # negative truth; NNLS must clip
    }
    samples = _synth_samples(300, true, noise=0.02, seed=7)
    result = calibrate(samples)

    for v in result.weights.values():
        assert v >= 0.0


def test_calibrate_rejects_mismatched_bloc_scales() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        calibrate([], momentum_bloc_scale=0.9, alpha_bloc_scale=0.2)


def test_calibrate_handles_empty_sample_list() -> None:
    result = calibrate([])
    assert result.sample_size == 0
    for v in result.weights.values():
        assert v == 0.0


def test_round_trip_artifact(tmp_path) -> None:
    true = {"momentum_1m": 0.5, "momentum_3m": 0.5}
    samples = _synth_samples(100, true, seed=1)
    weights = calibrate(samples)
    path = write_artifact(weights, tmp_path)
    reloaded = read_artifact(path)
    assert isinstance(reloaded, CalibratedWeights)
    assert reloaded.weights == weights.weights
    assert reloaded.sample_size == weights.sample_size


def test_calibrate_skips_rows_with_nan_forward_return() -> None:
    true = {"momentum_1m": 0.5}
    samples = _synth_samples(10, true, noise=0.0, seed=3)
    samples.append(
        CalibrationSample(
            as_of=datetime(2024, 6, 1, tzinfo=_UTC),
            instrument_id=uuid.uuid4(),
            features={name: 0.1 for name in list(MOMENTUM_BLOC) + list(ALPHA_BLOC)},
            forward_return=float("nan"),
        )
    )
    result = calibrate(samples)
    # Invalid row should not have been counted; sample_size reflects
    # the filtered count.
    assert result.sample_size == 10
