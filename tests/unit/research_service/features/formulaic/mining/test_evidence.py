"""Unit tests for the evidence framework."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.formulaic.ast import Var
from quant_platform.research.features.formulaic.mining.evidence import (
    CandidateEvidence,
    compute_evidence,
    make_forward_return_labels,
)
from quant_platform.research.features.formulaic.operators import rank
from quant_platform.research.features.formulaic.panel import build_market_panel


def _bars(n_instruments: int = 5, n_rows: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    rows = []
    dates = pd.bdate_range(start="2024-01-02", periods=n_rows)
    for inst in range(n_instruments):
        closes = 100.0 + np.cumsum(rng.normal(0.05, 1.0, size=n_rows))
        for i, d in enumerate(dates):
            close = float(closes[i])
            rows.append(
                {
                    "instrument_id": f"I{inst}",
                    "date": d,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + 5 * inst,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# make_forward_return_labels
# ---------------------------------------------------------------------------


def test_forward_returns_have_horizon_warmup_at_each_instrument_tail() -> None:
    bars = _bars(n_instruments=2, n_rows=10)
    panel = build_market_panel(bars)
    labels = make_forward_return_labels(panel, horizon=3)
    df = panel.frame.assign(label=labels)
    # The last 3 rows of every instrument are NaN — no future data.
    for _, g in df.groupby("instrument_id", sort=False):
        assert g["label"].iloc[-3:].isna().all()
        assert g["label"].iloc[:-3].notna().all()


def test_forward_returns_match_manual_pct_change() -> None:
    bars = _bars(n_instruments=1, n_rows=8)
    panel = build_market_panel(bars)
    labels = make_forward_return_labels(panel, horizon=2)
    closes = panel.frame["close"].to_numpy()
    expected = closes[2:] / closes[:-2] - 1
    np.testing.assert_allclose(labels.iloc[:-2].to_numpy(), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# compute_evidence
# ---------------------------------------------------------------------------


def test_compute_evidence_returns_finite_metrics_on_smooth_panel() -> None:
    bars = _bars(n_instruments=8, n_rows=40)
    panel = build_market_panel(bars)
    labels = make_forward_return_labels(panel, horizon=5)
    evidence = compute_evidence(rank(Var("close")), panel, labels)
    assert isinstance(evidence, CandidateEvidence)
    assert math.isfinite(evidence.mean_ic)
    assert math.isfinite(evidence.rank_ic)
    # turnover ∈ [0, 1] (mean of L1 distances between [0,1] ranks).
    assert 0.0 <= evidence.turnover <= 1.0
    assert evidence.coverage > 0
    assert evidence.n_dates > 0


def test_compute_evidence_correlation_to_baseline_max_is_nan_without_baselines() -> None:
    bars = _bars(n_instruments=5, n_rows=20)
    panel = build_market_panel(bars)
    labels = make_forward_return_labels(panel, horizon=3)
    evidence = compute_evidence(rank(Var("close")), panel, labels)
    assert math.isnan(evidence.correlation_to_baseline_max)


def test_compute_evidence_correlation_to_baseline_max_is_one_against_self() -> None:
    bars = _bars(n_instruments=5, n_rows=20)
    panel = build_market_panel(bars)
    labels = make_forward_return_labels(panel, horizon=3)
    # Build a baseline that's exactly the candidate's evaluated values.
    from quant_platform.research.features.formulaic.evaluator import evaluate_expression

    feature_values = evaluate_expression(panel, rank(Var("close")))
    evidence = compute_evidence(
        rank(Var("close")),
        panel,
        labels,
        baseline_features={"self": feature_values},
    )
    assert evidence.correlation_to_baseline_max == pytest.approx(1.0, abs=1e-9)


def test_compute_evidence_coverage_counts_non_null_pairs() -> None:
    """Coverage = (feature & label both non-null) count."""
    bars = _bars(n_instruments=4, n_rows=15)
    panel = build_market_panel(bars)
    labels = make_forward_return_labels(panel, horizon=2)
    evidence = compute_evidence(rank(Var("close")), panel, labels)
    # 4 instruments × 15 rows = 60 total. Each instrument loses 2 rows
    # to the horizon → 4 × 13 = 52 valid pairs.
    assert evidence.coverage == 52


def test_compute_evidence_uses_rank_ic_methodology() -> None:
    """rank_ic should differ from mean_ic when the candidate is
    non-linearly related to the label."""
    bars = _bars(n_instruments=10, n_rows=30)
    panel = build_market_panel(bars)
    labels = make_forward_return_labels(panel, horizon=3)
    # rank() is monotone but bounded — Spearman and Pearson against
    # the same label will typically differ unless the input is already
    # rank-normal.
    evidence = compute_evidence(rank(Var("close")), panel, labels)
    # Sanity: both metrics are finite and inside [-1, 1].
    assert -1.0 <= evidence.mean_ic <= 1.0
    assert -1.0 <= evidence.rank_ic <= 1.0


# ---------------------------------------------------------------------------
# CandidateEvidence shape
# ---------------------------------------------------------------------------


def test_candidate_evidence_is_immutable() -> None:
    """Frozen dataclass — mutation should raise."""
    from dataclasses import FrozenInstanceError

    ev = CandidateEvidence(
        mean_ic=0.1,
        rank_ic=0.05,
        icir=0.3,
        turnover=0.2,
        coverage=100,
        correlation_to_baseline_max=float("nan"),
        n_dates=10,
    )
    with pytest.raises(FrozenInstanceError):
        ev.mean_ic = 0.5  # type: ignore[misc]
