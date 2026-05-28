"""Unit tests for :class:`RandomSearch` and :class:`EvolutionarySearch`."""

from __future__ import annotations

import random

import pytest

from quant_platform.research.features.formulaic.ast import Expression
from quant_platform.research.features.formulaic.mining.grammar import AlphaGrammar
from quant_platform.research.features.formulaic.mining.search import (
    EvolutionarySearch,
    RandomSearch,
)


def _fitness_by_depth(expr: Expression) -> float:
    """Toy fitness: deeper trees score higher.

    Used to verify EvolutionarySearch's selection pressure produces a
    population whose best-fitness rises over generations.
    """
    return float(sum(1 for _ in expr.walk()))


# ---------------------------------------------------------------------------
# RandomSearch
# ---------------------------------------------------------------------------


def test_random_search_yields_requested_count() -> None:
    g = AlphaGrammar(max_depth=3, max_total_lookback=120)
    rng = random.Random(0)
    yielded = list(RandomSearch(n_candidates=10).iterate(g, rng, lambda _e: 0.0))
    assert len(yielded) == 10
    for expr, gen, parent, kind in yielded:
        assert gen == 0
        assert parent is None
        assert kind is None
        assert isinstance(expr, Expression)


def test_random_search_is_reproducible_with_seed() -> None:
    g = AlphaGrammar(max_depth=3, max_total_lookback=120)
    a = [
        expr
        for expr, *_ in RandomSearch(n_candidates=5).iterate(g, random.Random(99), lambda _e: 0.0)
    ]
    b = [
        expr
        for expr, *_ in RandomSearch(n_candidates=5).iterate(g, random.Random(99), lambda _e: 0.0)
    ]
    assert a == b


def test_random_search_rejects_zero_candidates() -> None:
    with pytest.raises(ValueError):
        RandomSearch(n_candidates=0)


# ---------------------------------------------------------------------------
# EvolutionarySearch
# ---------------------------------------------------------------------------


def test_evolutionary_search_validation() -> None:
    with pytest.raises(ValueError):
        EvolutionarySearch(population_size=0)
    with pytest.raises(ValueError):
        EvolutionarySearch(n_generations=-1)
    with pytest.raises(ValueError, match="elite_size"):
        EvolutionarySearch(population_size=10, elite_size=11)
    with pytest.raises(ValueError, match="tournament_size"):
        EvolutionarySearch(population_size=10, tournament_size=20)


def test_evolutionary_search_yields_full_population_count() -> None:
    g = AlphaGrammar(max_depth=3, max_total_lookback=120)
    search = EvolutionarySearch(population_size=8, n_generations=2, elite_size=2, tournament_size=2)
    yielded = list(search.iterate(g, random.Random(0), _fitness_by_depth))
    # 8 in gen 0 + 8 × 2 in subsequent generations = 24 yields.
    assert len(yielded) == 24


def test_evolutionary_search_records_generation_index() -> None:
    g = AlphaGrammar(max_depth=3, max_total_lookback=120)
    search = EvolutionarySearch(population_size=4, n_generations=3, elite_size=1, tournament_size=2)
    by_gen: dict[int, int] = {}
    for _expr, gen, _parent, _kind in search.iterate(g, random.Random(0), _fitness_by_depth):
        by_gen[gen] = by_gen.get(gen, 0) + 1
    assert by_gen == {0: 4, 1: 4, 2: 4, 3: 4}


def test_evolutionary_search_records_parent_for_non_seed_generations() -> None:
    g = AlphaGrammar(max_depth=3, max_total_lookback=120)
    search = EvolutionarySearch(population_size=4, n_generations=2, elite_size=1, tournament_size=2)
    has_parent: dict[int, list[bool]] = {}
    for _expr, gen, parent, _kind in search.iterate(g, random.Random(0), _fitness_by_depth):
        has_parent.setdefault(gen, []).append(parent is not None)
    # Seed gen has no parents; later gens have parents on every entry.
    assert all(not p for p in has_parent[0])
    for gen in (1, 2):
        assert all(has_parent[gen])


def test_evolutionary_search_marks_elites() -> None:
    g = AlphaGrammar(max_depth=3, max_total_lookback=120)
    search = EvolutionarySearch(population_size=6, n_generations=1, elite_size=2, tournament_size=2)
    elite_kinds: list[str | None] = []
    for _expr, gen, _parent, kind in search.iterate(g, random.Random(0), _fitness_by_depth):
        if gen == 1:
            elite_kinds.append(kind)
    # First 2 entries in gen 1 are elites; remainder are mutated.
    assert elite_kinds[:2] == ["elite", "elite"]
    assert all(k != "elite" for k in elite_kinds[2:])


def test_evolutionary_search_best_fitness_rises_with_depth_pressure() -> None:
    """With ``_fitness_by_depth`` rewarding deeper trees, evolutionary
    search should produce a population whose best-fitness in the final
    generation is at least as high as in generation 0."""
    g = AlphaGrammar(max_depth=4, max_total_lookback=120)
    search = EvolutionarySearch(
        population_size=12, n_generations=3, elite_size=2, tournament_size=3
    )
    by_gen_best: dict[int, float] = {}
    for expr, gen, _parent, _kind in search.iterate(g, random.Random(0), _fitness_by_depth):
        fitness = _fitness_by_depth(expr)
        by_gen_best[gen] = max(by_gen_best.get(gen, float("-inf")), fitness)
    # Elitism guarantees monotone non-decrease across generations.
    assert by_gen_best[3] >= by_gen_best[0]


def test_evolutionary_search_is_reproducible_with_seed() -> None:
    g = AlphaGrammar(max_depth=3, max_total_lookback=120)
    search = EvolutionarySearch(population_size=6, n_generations=2, elite_size=1, tournament_size=2)

    def _run() -> list[Expression]:
        return [expr for expr, *_ in search.iterate(g, random.Random(7), _fitness_by_depth)]

    assert _run() == _run()
