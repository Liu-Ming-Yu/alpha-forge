"""Unit tests for the admission gate."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from quant_platform.research.features.formulaic.ast import Var
from quant_platform.research.features.formulaic.mining.admission import (
    AdmissionGate,
    AdmissionThresholds,
)
from quant_platform.research.features.formulaic.mining.evidence import CandidateEvidence
from quant_platform.research.features.formulaic.mining.provenance import (
    AutoAlphaProvenance,
)
from quant_platform.research.features.formulaic.operators import rank


def _provenance(
    *,
    rank_ic: float = 0.05,
    icir: float = 0.3,
    turnover: float = 0.2,
    coverage: int = 500,
    n_dates: int = 50,
    correlation: float = float("nan"),
) -> AutoAlphaProvenance:
    return AutoAlphaProvenance(
        name="auto_alpha_000001",
        expression=rank(Var("close")),
        generation=0,
        seed=42,
        parent_name=None,
        mutation_kind=None,
        operator_set_version="operator-set-v1",
        feature_set_version="formulaic-alpha-v1",
        evidence=CandidateEvidence(
            mean_ic=0.05,
            rank_ic=rank_ic,
            icir=icir,
            turnover=turnover,
            coverage=coverage,
            correlation_to_baseline_max=correlation,
            n_dates=n_dates,
        ),
        created_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# AdmissionThresholds
# ---------------------------------------------------------------------------


def test_admission_thresholds_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="min_rank_ic"):
        AdmissionThresholds(min_rank_ic=2.0)
    with pytest.raises(ValueError, match="max_turnover"):
        AdmissionThresholds(max_turnover=0.0)
    with pytest.raises(ValueError, match="min_coverage_ratio"):
        AdmissionThresholds(min_coverage_ratio=1.5)
    with pytest.raises(ValueError, match="max_correlation_to_admitted"):
        AdmissionThresholds(max_correlation_to_admitted=-0.1)
    with pytest.raises(ValueError, match="min_n_dates"):
        AdmissionThresholds(min_n_dates=0)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_admission_gate_admits_when_every_check_passes() -> None:
    gate = AdmissionGate()
    decision = gate.evaluate(_provenance(), [], panel_size=1000)
    assert decision.admitted is True
    assert decision.reason == "admitted"
    assert decision.failed_checks == ()


def test_admission_gate_rejects_low_rank_ic() -> None:
    gate = AdmissionGate(thresholds=AdmissionThresholds(min_rank_ic=0.05))
    decision = gate.evaluate(_provenance(rank_ic=0.01), [], panel_size=1000)
    assert decision.admitted is False
    assert "rank_ic" in decision.failed_checks


def test_admission_gate_rejects_nan_rank_ic() -> None:
    gate = AdmissionGate()
    decision = gate.evaluate(_provenance(rank_ic=float("nan")), [], panel_size=1000)
    assert decision.admitted is False
    assert "rank_ic" in decision.failed_checks


def test_admission_gate_rejects_high_turnover() -> None:
    gate = AdmissionGate(thresholds=AdmissionThresholds(max_turnover=0.1))
    decision = gate.evaluate(_provenance(turnover=0.5), [], panel_size=1000)
    assert decision.admitted is False
    assert "turnover" in decision.failed_checks


def test_admission_gate_rejects_low_icir() -> None:
    gate = AdmissionGate(thresholds=AdmissionThresholds(min_icir=0.5))
    decision = gate.evaluate(_provenance(icir=0.1), [], panel_size=1000)
    assert decision.admitted is False
    assert "icir" in decision.failed_checks


def test_admission_gate_rejects_low_coverage_ratio() -> None:
    gate = AdmissionGate(thresholds=AdmissionThresholds(min_coverage_ratio=0.9))
    decision = gate.evaluate(_provenance(coverage=100), [], panel_size=1000)
    assert decision.admitted is False
    assert "coverage_ratio" in decision.failed_checks


def test_admission_gate_rejects_low_n_dates() -> None:
    gate = AdmissionGate(thresholds=AdmissionThresholds(min_n_dates=100))
    decision = gate.evaluate(_provenance(n_dates=30), [], panel_size=1000)
    assert decision.admitted is False
    assert "n_dates" in decision.failed_checks


def test_admission_gate_rejects_high_correlation_to_baseline() -> None:
    gate = AdmissionGate(thresholds=AdmissionThresholds(max_correlation_to_admitted=0.5))
    decision = gate.evaluate(
        _provenance(correlation=0.9),
        [],
        panel_size=1000,
    )
    assert decision.admitted is False
    assert "correlation_to_admitted" in decision.failed_checks


def test_admission_gate_tolerates_nan_correlation_when_no_baseline() -> None:
    """No baseline → NaN correlation → that check should not fire."""
    gate = AdmissionGate()
    decision = gate.evaluate(
        _provenance(correlation=float("nan")),
        [],
        panel_size=1000,
    )
    assert decision.admitted is True
    assert "correlation_to_admitted" not in decision.failed_checks


def test_admission_gate_reports_every_failing_check() -> None:
    """When multiple checks fail, all of them are surfaced."""
    gate = AdmissionGate(
        thresholds=AdmissionThresholds(min_rank_ic=0.1, min_icir=0.5, max_turnover=0.05)
    )
    decision = gate.evaluate(
        _provenance(rank_ic=0.01, icir=0.1, turnover=0.3),
        [],
        panel_size=1000,
    )
    assert decision.admitted is False
    assert set(decision.failed_checks) >= {"rank_ic", "icir", "turnover"}


def test_admission_decision_reason_is_human_readable() -> None:
    gate = AdmissionGate(thresholds=AdmissionThresholds(min_rank_ic=0.05))
    decision = gate.evaluate(_provenance(rank_ic=0.01), [], panel_size=1000)
    assert "rank_ic" in decision.reason
    assert "<" in decision.reason
    # The threshold value should be visible.
    assert "0.05" in decision.reason


def test_panel_size_zero_skips_coverage_check() -> None:
    """Zero panel size means the ratio is undefined — skip the check
    instead of dividing by zero."""
    gate = AdmissionGate()
    decision = gate.evaluate(_provenance(coverage=0), [], panel_size=0)
    # Other checks may still pass; the coverage_ratio one is skipped.
    assert "coverage_ratio" not in decision.failed_checks


# ---------------------------------------------------------------------------
# AdmissionDecision sanity
# ---------------------------------------------------------------------------


def test_admission_decision_admitted_carries_no_failed_checks() -> None:
    gate = AdmissionGate()
    decision = gate.evaluate(_provenance(), [], panel_size=1000)
    assert decision.admitted is True
    assert decision.failed_checks == ()
    # NaN values shouldn't leak into "admitted=True" outcomes.
    assert not math.isnan(decision.admitted)  # type: ignore[arg-type]
