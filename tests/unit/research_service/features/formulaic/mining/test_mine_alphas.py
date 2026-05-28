"""End-to-end tests for :func:`mine_alphas`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.formulaic.mining import (
    AdmissionGate,
    AdmissionThresholds,
    AlphaGrammar,
    AutoAlphaProvenance,
    EvolutionarySearch,
    MiningResult,
    RandomSearch,
    make_forward_return_labels,
    mine_alphas,
)
from quant_platform.research.features.formulaic.panel import build_market_panel


def _wide_bars(n_instruments: int = 8, n_rows: int = 100) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_mine_alphas_runs_end_to_end_with_random_search() -> None:
    panel = build_market_panel(_wide_bars())
    labels = make_forward_return_labels(panel, horizon=5)
    grammar = AlphaGrammar(max_depth=3, max_total_lookback=60)
    result = mine_alphas(
        grammar=grammar,
        panel=panel,
        labels=labels,
        search=RandomSearch(n_candidates=15),
        gate=AdmissionGate(),
        seed=42,
    )
    assert isinstance(result, MiningResult)
    assert result.n_evaluated == 15
    assert len(result.history) == 15
    # admitted is a subset of history.
    assert set(p.name for p in result.admitted) <= {p.name for p in result.history}


def test_mine_alphas_is_reproducible_with_seed() -> None:
    panel = build_market_panel(_wide_bars())
    labels = make_forward_return_labels(panel, horizon=5)
    grammar = AlphaGrammar(max_depth=3, max_total_lookback=60)

    def _run() -> MiningResult:
        return mine_alphas(
            grammar=grammar,
            panel=panel,
            labels=labels,
            search=RandomSearch(n_candidates=10),
            gate=AdmissionGate(),
            seed=123,
        )

    a, b = _run(), _run()
    # Identical histories (names, expressions, evidence, admission).
    # Field-by-field comparison with NaN-awareness: pandas' float-NaN
    # comparisons return False even for "the same NaN", so we can't
    # rely on dataclass ``==`` to compare evidence rows.
    assert len(a.history) == len(b.history)
    for pa, pb in zip(a.history, b.history, strict=True):
        assert pa.expression == pb.expression
        assert pa.admitted == pb.admitted
        _assert_evidence_equal(pa.evidence, pb.evidence)


def _assert_evidence_equal(a, b) -> None:  # type: ignore[no-untyped-def]
    """NaN-aware structural equality for :class:`CandidateEvidence`."""
    import math

    for field_name in (
        "mean_ic",
        "rank_ic",
        "icir",
        "turnover",
        "correlation_to_baseline_max",
    ):
        av = getattr(a, field_name)
        bv = getattr(b, field_name)
        if math.isnan(av):
            assert math.isnan(bv), f"{field_name}: a=NaN but b={bv}"
        else:
            assert av == pytest.approx(bv), f"{field_name}: {av} != {bv}"
    assert a.coverage == b.coverage
    assert a.n_dates == b.n_dates


def test_mine_alphas_runs_with_evolutionary_search() -> None:
    panel = build_market_panel(_wide_bars(n_instruments=6, n_rows=80))
    labels = make_forward_return_labels(panel, horizon=5)
    grammar = AlphaGrammar(max_depth=3, max_total_lookback=60)
    result = mine_alphas(
        grammar=grammar,
        panel=panel,
        labels=labels,
        search=EvolutionarySearch(
            population_size=6, n_generations=2, elite_size=1, tournament_size=2
        ),
        gate=AdmissionGate(thresholds=AdmissionThresholds(min_rank_ic=-0.5, min_icir=-2.0)),
        seed=7,
    )
    # 6 in gen 0 + 6 in gen 1 + 6 in gen 2 = 18 evaluations.
    assert result.n_evaluated == 18
    # Some generation must have a non-zero index.
    assert {p.generation for p in result.history} == {0, 1, 2}


# ---------------------------------------------------------------------------
# Provenance contents
# ---------------------------------------------------------------------------


def test_provenance_carries_required_fields() -> None:
    panel = build_market_panel(_wide_bars(n_instruments=4, n_rows=40))
    labels = make_forward_return_labels(panel, horizon=3)
    result = mine_alphas(
        grammar=AlphaGrammar(max_depth=2, max_total_lookback=30),
        panel=panel,
        labels=labels,
        search=RandomSearch(n_candidates=4),
        gate=AdmissionGate(),
        seed=11,
    )
    for prov in result.history:
        assert isinstance(prov, AutoAlphaProvenance)
        assert prov.name.startswith("auto_alpha_")
        assert prov.seed == 11
        assert prov.operator_set_version == "operator-set-v1"
        assert prov.feature_set_version == "formulaic-alpha-v1"
        assert prov.evidence is not None
        # Admission reason is always set (either "admitted" or a
        # description of which check failed).
        assert prov.admission_reason


def test_provenance_records_parent_for_evolutionary_children() -> None:
    """Non-seed generations have parent_name set; seed gen does not."""
    panel = build_market_panel(_wide_bars(n_instruments=4, n_rows=40))
    labels = make_forward_return_labels(panel, horizon=3)
    result = mine_alphas(
        grammar=AlphaGrammar(max_depth=2, max_total_lookback=30),
        panel=panel,
        labels=labels,
        search=EvolutionarySearch(
            population_size=4, n_generations=1, elite_size=1, tournament_size=2
        ),
        gate=AdmissionGate(thresholds=AdmissionThresholds(min_rank_ic=-1.0)),
        seed=13,
    )
    seed_provs = [p for p in result.history if p.generation == 0]
    child_provs = [p for p in result.history if p.generation == 1]
    assert all(p.parent_name is None for p in seed_provs)
    assert all(p.parent_name is not None for p in child_provs)
    # Mutation kind is set on non-elite children (elites have "elite";
    # mutated children get one of the MUTATION_KINDS).
    assert all(p.mutation_kind is not None for p in child_provs)


# ---------------------------------------------------------------------------
# Correlation pruning
# ---------------------------------------------------------------------------


def test_running_baseline_grows_as_admissions_happen() -> None:
    """After mining, each admitted alpha after the first should have
    had a non-empty baseline available during its correlation check —
    i.e. ``correlation_to_baseline_max`` is finite for the 2nd, 3rd,
    … admissions."""
    panel = build_market_panel(_wide_bars(n_instruments=8, n_rows=80))
    labels = make_forward_return_labels(panel, horizon=5)
    grammar = AlphaGrammar(max_depth=3, max_total_lookback=60)
    # Use a permissive gate so we get >=2 admissions for the test.
    result = mine_alphas(
        grammar=grammar,
        panel=panel,
        labels=labels,
        search=RandomSearch(n_candidates=30),
        gate=AdmissionGate(
            thresholds=AdmissionThresholds(
                min_rank_ic=-1.0,
                min_icir=-10.0,
                max_turnover=1.0,
                min_coverage_ratio=0.0,
                min_n_dates=1,
                max_correlation_to_admitted=1.0,
            )
        ),
        seed=17,
    )
    admitted = result.admitted
    if len(admitted) < 2:
        pytest.skip("Not enough admissions on the seed to exercise the running-baseline path")
    # The first admission had no baseline; every subsequent admission
    # should have had a finite correlation_to_baseline_max recorded.
    # (We can't recompute the at-admission baseline here, but the
    # invariant the test exercises is "the running baseline is wired
    # into the evidence path" — verified by the existence of finite
    # corr values on any post-first admission.)
    from math import isnan

    finite_corr_post_first = any(
        not isnan(p.evidence.correlation_to_baseline_max) for p in admitted[1:]
    )
    assert finite_corr_post_first
