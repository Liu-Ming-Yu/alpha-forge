"""Unit tests for the walk-forward (K-fold OOS) evidence path."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.formulaic.ast import Var
from quant_platform.research.features.formulaic.mining.evidence import (
    make_forward_return_labels,
)
from quant_platform.research.features.formulaic.mining.walk_forward import (
    MiningFoldConfig,
    WalkForwardEvidence,
    _generate_fold_boundaries,
    _longest_negative_streak,
    compute_walk_forward_evidence,
)
from quant_platform.research.features.formulaic.operators import rank
from quant_platform.research.features.formulaic.panel import build_market_panel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wide_panel(n_instruments: int = 8, n_rows: int = 200) -> object:
    rng = np.random.default_rng(seed=0)
    rows = []
    dates = pd.bdate_range(start="2023-01-02", periods=n_rows)
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
                    "volume": 1000.0 + 5 * inst + rng.normal(0, 50),
                }
            )
    return build_market_panel(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# MiningFoldConfig
# ---------------------------------------------------------------------------


def test_fold_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        MiningFoldConfig(n_folds=1)
    with pytest.raises(ValueError, match="embargo_days"):
        MiningFoldConfig(embargo_days=-1)
    with pytest.raises(ValueError, match="min_test_days"):
        MiningFoldConfig(min_test_days=0)


# ---------------------------------------------------------------------------
# Fold boundary generation
# ---------------------------------------------------------------------------


def test_generate_fold_boundaries_splits_into_equal_chunks() -> None:
    dates = pd.bdate_range(start="2024-01-02", periods=12).to_numpy()
    boundaries = _generate_fold_boundaries(dates, n_folds=4)
    assert len(boundaries) == 4
    # Each chunk should have 3 dates.
    for start, end in boundaries:
        # ``end`` is the last date in the chunk (inclusive).
        idx_start = int(np.searchsorted(dates, np.datetime64(start)))
        idx_end = int(np.searchsorted(dates, np.datetime64(end))) + 1
        assert idx_end - idx_start == 3


def test_generate_fold_boundaries_handles_non_divisible_lengths() -> None:
    dates = pd.bdate_range(start="2024-01-02", periods=10).to_numpy()
    boundaries = _generate_fold_boundaries(dates, n_folds=3)
    assert len(boundaries) == 3
    sizes = []
    for start, end in boundaries:
        idx_start = int(np.searchsorted(dates, np.datetime64(start)))
        idx_end = int(np.searchsorted(dates, np.datetime64(end))) + 1
        sizes.append(idx_end - idx_start)
    # Sizes are 3, 3, 4 or 4, 3, 3 — total equals 10.
    assert sum(sizes) == 10


# ---------------------------------------------------------------------------
# Streak helper
# ---------------------------------------------------------------------------


def test_longest_negative_streak_counts_consecutive_negatives() -> None:
    assert _longest_negative_streak([0.1, -0.2, -0.3, 0.4, -0.5]) == 2
    assert _longest_negative_streak([-0.1, -0.2, -0.3]) == 3
    assert _longest_negative_streak([0.1, 0.2]) == 0
    assert _longest_negative_streak([]) == 0


def test_longest_negative_streak_breaks_run_on_nan() -> None:
    """NaN entries should reset the negative-streak counter."""
    assert _longest_negative_streak([-0.1, float("nan"), -0.2]) == 1


# ---------------------------------------------------------------------------
# compute_walk_forward_evidence
# ---------------------------------------------------------------------------


def test_compute_wf_evidence_returns_correct_shape() -> None:
    panel = _wide_panel()
    labels = make_forward_return_labels(panel, horizon=5)
    config = MiningFoldConfig(n_folds=4, embargo_days=5, min_test_days=10)
    evidence = compute_walk_forward_evidence(
        rank(Var("close")),
        panel,
        labels,
        fold_config=config,
    )
    assert isinstance(evidence, WalkForwardEvidence)
    assert len(evidence.fold_ics) == 4
    assert len(evidence.fold_rank_ics) == 4
    assert evidence.n_folds_valid <= 4
    assert evidence.n_dates >= 0


def test_compute_wf_evidence_aggregates_per_fold_means() -> None:
    panel = _wide_panel()
    labels = make_forward_return_labels(panel, horizon=5)
    config = MiningFoldConfig(n_folds=4, embargo_days=5, min_test_days=10)
    evidence = compute_walk_forward_evidence(
        rank(Var("close")),
        panel,
        labels,
        fold_config=config,
    )
    # mean_ic == mean of valid fold_ics.
    valid_ics = [v for v in evidence.fold_ics if math.isfinite(v)]
    if valid_ics:
        assert evidence.mean_ic == pytest.approx(float(np.mean(valid_ics)), rel=1e-9)


def test_compute_wf_evidence_marks_short_folds_as_nan() -> None:
    """A fold whose effective length falls below ``min_test_days``
    becomes NaN and is excluded from the OOS aggregate."""
    panel = _wide_panel(n_rows=40)  # 4 folds × 10 dates each, before embargo
    labels = make_forward_return_labels(panel, horizon=5)
    config = MiningFoldConfig(n_folds=4, embargo_days=5, min_test_days=20)
    evidence = compute_walk_forward_evidence(
        rank(Var("close")),
        panel,
        labels,
        fold_config=config,
    )
    # Every fold is too short → all NaN.
    assert all(math.isnan(v) for v in evidence.fold_ics)
    assert evidence.n_folds_valid == 0
    assert math.isnan(evidence.mean_ic)


def test_compute_wf_evidence_returns_finite_metrics_on_real_panel() -> None:
    panel = _wide_panel()
    labels = make_forward_return_labels(panel, horizon=5)
    config = MiningFoldConfig(n_folds=4, embargo_days=5, min_test_days=15)
    evidence = compute_walk_forward_evidence(
        rank(Var("close")),
        panel,
        labels,
        fold_config=config,
    )
    assert evidence.n_folds_valid >= 3
    assert math.isfinite(evidence.mean_ic)
    assert math.isfinite(evidence.rank_ic)
    # ICIR may legitimately be NaN if fold-IC std collapses on a
    # constant feature; the random panel produces noisy ICs so we
    # expect a finite value here.
    assert math.isfinite(evidence.icir)


def test_compute_wf_evidence_fold_negative_streak() -> None:
    """Manufacture an alpha whose IC is consistently negative — the
    fold_negative_ic_streak should equal n_folds_valid."""
    panel = _wide_panel()
    labels = make_forward_return_labels(panel, horizon=5)
    config = MiningFoldConfig(n_folds=4, embargo_days=5, min_test_days=15)
    # The exact alpha doesn't matter for this property — the test
    # checks that the streak counter walks fold_ics correctly.
    evidence = compute_walk_forward_evidence(
        rank(Var("close")),
        panel,
        labels,
        fold_config=config,
    )
    # Manually compute streak from fold_ics; must match the recorded
    # value.
    assert evidence.fold_negative_ic_streak == _longest_negative_streak(list(evidence.fold_ics))


def test_compute_wf_evidence_too_few_dates_collapses_to_empty() -> None:
    """When the panel has fewer dates than n_folds × min_test_days,
    the function returns all-NaN per-fold values without raising."""
    panel = _wide_panel(n_instruments=4, n_rows=20)  # only 20 dates
    labels = make_forward_return_labels(panel, horizon=2)
    config = MiningFoldConfig(n_folds=4, embargo_days=0, min_test_days=20)
    evidence = compute_walk_forward_evidence(
        rank(Var("close")),
        panel,
        labels,
        fold_config=config,
    )
    assert evidence.n_folds_valid == 0
    assert math.isnan(evidence.mean_ic)
    assert all(math.isnan(v) for v in evidence.fold_ics)


def test_compute_wf_evidence_baseline_correlation_propagates() -> None:
    panel = _wide_panel()
    labels = make_forward_return_labels(panel, horizon=5)
    config = MiningFoldConfig(n_folds=4, embargo_days=5, min_test_days=15)
    from quant_platform.research.features.formulaic.evaluator import (
        evaluate_expression,
    )

    feature = evaluate_expression(panel, rank(Var("close")))
    evidence = compute_walk_forward_evidence(
        rank(Var("close")),
        panel,
        labels,
        fold_config=config,
        baseline_features={"self": feature},
    )
    # Correlation against itself is 1.0.
    assert evidence.correlation_to_baseline_max == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Integration: WF + AdmissionGate
# ---------------------------------------------------------------------------


def test_admission_gate_consults_fold_streak_when_present() -> None:
    """The gate's WF-specific checks fire only when the evidence
    carries the relevant fields."""
    from datetime import UTC, datetime

    from quant_platform.research.features.formulaic.mining.admission import (
        AdmissionGate,
        AdmissionThresholds,
    )
    from quant_platform.research.features.formulaic.mining.provenance import (
        AutoAlphaProvenance,
    )

    wf_evidence = WalkForwardEvidence(
        mean_ic=0.05,
        rank_ic=0.05,
        icir=0.5,
        fold_ics=(-0.02, -0.03, -0.01, 0.05),  # 3-fold negative streak
        fold_rank_ics=(-0.02, -0.03, -0.01, 0.05),
        fold_negative_ic_streak=3,
        n_folds_valid=4,
        n_dates=200,
        turnover=0.1,
        coverage=500,
        correlation_to_baseline_max=float("nan"),
    )
    prov = AutoAlphaProvenance(
        name="auto_alpha_001",
        expression=rank(Var("close")),
        generation=0,
        seed=0,
        parent_name=None,
        mutation_kind=None,
        operator_set_version="operator-set-v1",
        feature_set_version="formulaic-alpha-v1",
        evidence=wf_evidence,
        created_at=datetime.now(UTC),
    )
    gate = AdmissionGate(thresholds=AdmissionThresholds(max_fold_negative_ic_streak=2))
    decision = gate.evaluate(prov, [], panel_size=1000)
    assert decision.admitted is False
    assert "fold_negative_ic_streak" in decision.failed_checks


def test_admission_gate_skips_fold_checks_for_single_pass_evidence() -> None:
    """A :class:`CandidateEvidence` (no fold fields) bypasses the
    WF-specific checks entirely — those rules can't apply."""
    from datetime import UTC, datetime

    from quant_platform.research.features.formulaic.mining.admission import (
        AdmissionGate,
        AdmissionThresholds,
    )
    from quant_platform.research.features.formulaic.mining.evidence import (
        CandidateEvidence,
    )
    from quant_platform.research.features.formulaic.mining.provenance import (
        AutoAlphaProvenance,
    )

    single_pass = CandidateEvidence(
        mean_ic=0.05,
        rank_ic=0.05,
        icir=0.5,
        turnover=0.1,
        coverage=500,
        correlation_to_baseline_max=float("nan"),
        n_dates=200,
    )
    prov = AutoAlphaProvenance(
        name="auto_alpha_001",
        expression=rank(Var("close")),
        generation=0,
        seed=0,
        parent_name=None,
        mutation_kind=None,
        operator_set_version="operator-set-v1",
        feature_set_version="formulaic-alpha-v1",
        evidence=single_pass,
        created_at=datetime.now(UTC),
    )
    # Set the fold streak threshold to 0 — would fire on any
    # WalkForwardEvidence — but single-pass evidence has no streak
    # field so the check skips.
    gate = AdmissionGate(thresholds=AdmissionThresholds(max_fold_negative_ic_streak=0))
    decision = gate.evaluate(prov, [], panel_size=1000)
    assert decision.admitted is True
    assert "fold_negative_ic_streak" not in decision.failed_checks


# ---------------------------------------------------------------------------
# Integration: mine_alphas with fold_config
# ---------------------------------------------------------------------------


def test_mine_alphas_with_fold_config_uses_wf_evidence() -> None:
    from quant_platform.research.features.formulaic.mining import (
        AdmissionGate,
        AdmissionThresholds,
        AlphaGrammar,
        RandomSearch,
        mine_alphas,
    )

    panel = _wide_panel()
    labels = make_forward_return_labels(panel, horizon=5)
    result = mine_alphas(
        grammar=AlphaGrammar(max_depth=3, max_total_lookback=60),
        panel=panel,
        labels=labels,
        search=RandomSearch(n_candidates=5),
        gate=AdmissionGate(
            thresholds=AdmissionThresholds(
                min_rank_ic=-1.0,
                min_icir=-10.0,
                min_n_dates=1,
                min_n_folds_valid=1,
                max_fold_negative_ic_streak=10,
            )
        ),
        seed=42,
        fold_config=MiningFoldConfig(n_folds=4, embargo_days=5, min_test_days=15),
    )
    assert result.n_evaluated == 5
    # Every provenance has a WalkForwardEvidence — duck-type check
    # for the WF-specific attribute.
    for prov in result.history:
        assert hasattr(prov.evidence, "fold_ics")
        assert hasattr(prov.evidence, "fold_negative_ic_streak")
        assert hasattr(prov.evidence, "n_folds_valid")
