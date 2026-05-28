"""Tests for the IC and alpha-decay report."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quant_platform.services.research_service.reports.ic_report import (
    ICPanel,
    _average_ranks,
    _spearman,
    compute_ic_report,
    read_ic_report,
    write_ic_report,
)

_UTC = UTC


def test_average_ranks_handles_ties() -> None:
    ranks = _average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))
    # The two tied values should get the mean rank 1.5
    assert ranks.tolist() == [0.0, 1.5, 1.5, 3.0]


def test_spearman_detects_monotone_relationship() -> None:
    a = [1, 2, 3, 4, 5]
    b = [10, 20, 30, 40, 50]
    assert _spearman(a, b) == pytest.approx(1.0)


def test_spearman_nan_when_all_equal() -> None:
    assert np.isnan(_spearman([1, 1, 1], [4, 5, 6]))


def test_spearman_handles_too_few_rows() -> None:
    assert np.isnan(_spearman([1.0], [2.0]))


def _panel(as_of: datetime, factor: str, n: int, correlation: float) -> ICPanel:
    rng = np.random.default_rng(0)
    ids = [uuid.uuid4() for _ in range(n)]
    scores = rng.normal(size=n)
    noise = rng.normal(size=n) * (1.0 - abs(correlation))
    returns = correlation * scores + noise
    features = {iid: {factor: float(s)} for iid, s in zip(ids, scores, strict=False)}
    fwd = {iid: float(r) for iid, r in zip(ids, returns, strict=False)}
    return ICPanel(as_of=as_of, features=features, forward_returns=fwd)


def test_compute_ic_report_yields_per_factor_series() -> None:
    start = datetime(2024, 1, 1, tzinfo=_UTC)
    panels = [
        _panel(start + timedelta(days=i), "momentum_1m", n=30, correlation=0.8) for i in range(25)
    ]
    report = compute_ic_report(
        run_id=uuid.uuid4(),
        panels_by_horizon={1: panels, 5: panels, 10: panels, 20: panels},
        factors=["momentum_1m"],
    )
    assert len(report.series) == 1
    series = report.series[0]
    assert series.factor == "momentum_1m"
    assert len(series.ic) == 25

    # High-correlation synthetic data should produce positive rolling mean
    finite_rolling = [x for x in series.rolling_mean_20 if np.isfinite(x)]
    assert all(x > 0.3 for x in finite_rolling)


def test_round_trip_ic_report_artifact(tmp_path) -> None:
    start = datetime(2024, 1, 1, tzinfo=_UTC)
    panels = [
        _panel(start + timedelta(days=i), "momentum_1m", n=20, correlation=0.5) for i in range(5)
    ]
    report = compute_ic_report(
        run_id=uuid.uuid4(),
        panels_by_horizon={1: panels, 5: panels},
        factors=["momentum_1m"],
    )
    path = write_ic_report(report, tmp_path)
    reloaded = read_ic_report(path)
    assert reloaded["horizons"] == [1, 5]
    assert reloaded["series"][0]["factor"] == "momentum_1m"
    assert len(reloaded["series"][0]["ic"]) == 5
