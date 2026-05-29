"""Tests for qlib-style policy-guided alpha search."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.formulaic.ast import Expression
from quant_platform.research.features.formulaic.mining import (
    AdmissionGate,
    AdmissionThresholds,
    AlphaGrammar,
    make_forward_return_labels,
    mine_alphas,
)
from quant_platform.research.features.formulaic.mining.mutation import MUTATION_KINDS
from quant_platform.research.features.formulaic.mining.policy_search import (
    AlphaSearchActionInterpreter,
    AlphaSearchPolicyAction,
    PolicySearch,
)
from quant_platform.research.features.formulaic.panel import build_market_panel


def _fitness_by_nodes(expr: Expression) -> float:
    return float(sum(1 for _ in expr.walk()))


def _bars(n_instruments: int = 5, n_rows: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed=10)
    rows = []
    dates = pd.bdate_range(start="2024-01-02", periods=n_rows)
    for inst in range(n_instruments):
        closes = 100.0 + np.cumsum(rng.normal(0.02, 0.5, size=n_rows))
        for index, date in enumerate(dates):
            close = float(closes[index])
            rows.append(
                {
                    "instrument_id": f"I{inst}",
                    "date": date,
                    "open": close - 0.25,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0 + 10.0 * inst,
                }
            )
    return pd.DataFrame(rows)


def test_policy_search_yields_requested_count_and_lineage() -> None:
    grammar = AlphaGrammar(max_depth=3, max_total_lookback=60)
    yielded = list(
        PolicySearch(n_candidates=7).iterate(grammar, random.Random(4), _fitness_by_nodes)
    )

    assert len(yielded) == 7
    seed_expr, seed_generation, seed_parent, seed_kind = yielded[0]
    assert isinstance(seed_expr, Expression)
    assert seed_generation == 0
    assert seed_parent is None
    assert seed_kind is None
    for _expr, generation, parent, kind in yielded[1:]:
        assert generation > 0
        assert parent is not None
        assert kind in MUTATION_KINDS


def test_policy_search_is_reproducible_with_seed() -> None:
    grammar = AlphaGrammar(max_depth=3, max_total_lookback=60)
    search = PolicySearch(n_candidates=6)

    def _run() -> list[Expression]:
        return [expr for expr, *_ in search.iterate(grammar, random.Random(99), _fitness_by_nodes)]

    assert _run() == _run()


def test_policy_search_rejects_invalid_candidate_count() -> None:
    with pytest.raises(ValueError, match="n_candidates"):
        PolicySearch(n_candidates=0)


def test_policy_search_action_interpreter_rejects_unknown_mutation() -> None:
    interpreter = AlphaSearchActionInterpreter()
    with pytest.raises(ValueError, match="unknown mutation kind"):
        interpreter.validate_action(AlphaSearchPolicyAction("unknown"))


def test_policy_search_runs_through_mine_alphas() -> None:
    panel = build_market_panel(_bars())
    labels = make_forward_return_labels(panel, horizon=3)
    result = mine_alphas(
        grammar=AlphaGrammar(max_depth=2, max_total_lookback=30),
        panel=panel,
        labels=labels,
        search=PolicySearch(n_candidates=5),
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
        seed=22,
    )

    assert result.n_evaluated == 5
    assert [p.generation for p in result.history] == [0, 1, 2, 3, 4]
    assert result.history[0].mutation_kind is None
    assert all(p.mutation_kind in MUTATION_KINDS for p in result.history[1:])
