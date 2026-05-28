"""Search algorithms for the formulaic alpha miner.

The brief lists "genetic programming or reinforcement-learning style
search" as the target. This module ships the two simplest end-points
of that spectrum:

* :class:`RandomSearch` — generate ``n_candidates`` independent
  random ASTs from the grammar. Embarrassingly parallel; useful as a
  baseline and for fresh-seed diversity.
* :class:`EvolutionarySearch` — tournament-selection genetic
  programming with mutation but **no crossover** (subtree crossover
  on PIT-strict ASTs is fiddly enough that it deserves a separate
  sprint). Maintains a population of size ``population_size`` over
  ``n_generations``; each generation, the top-K survive untouched
  (elitism) and the rest are produced by mutating tournament winners.

Both algorithms expose ``iterate(grammar, rng, fitness_fn)`` which
yields ``(expression, generation, parent_name, mutation_kind)`` tuples
in the order the miner driver should evaluate them. The driver
(:func:`..__init__.mine_alphas`) is in charge of running the actual
:func:`..evidence.compute_evidence` and wrapping each evaluation in
an :class:`AutoAlphaProvenance`.

Parent tracking
---------------

For :class:`RandomSearch`, ``parent_name`` is always ``None`` and
``mutation_kind`` is always ``None`` — every candidate is a fresh
sample.

For :class:`EvolutionarySearch`, the children of one generation
record the *previous* generation's candidate name as their parent.
The driver uses that linkage to assemble the lineage tree after the
run completes.

Fitness function
----------------

Searchers don't compute fitness themselves. The driver passes a
``fitness_fn(expression) -> float`` callable. This decouples search
from evaluation: a caller running quick smoke tests can use a fast
mock fitness; production runs use
:func:`..evidence.compute_evidence` wrapped to return ``rank_ic``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from quant_platform.research.features.formulaic.mining.mutation import mutate

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterator

    from quant_platform.research.features.formulaic.ast import Expression
    from quant_platform.research.features.formulaic.mining.grammar import AlphaGrammar


#: Tuple yielded by every :class:`SearchAlgorithm.iterate` step.
#: Documented as a named tuple-of-strings here so callers can read
#: the search loop without consulting field positions.
SearchYield = tuple["Expression", int, str | None, str | None]


@runtime_checkable
class SearchAlgorithm(Protocol):
    """Search-loop contract.

    Implementations yield candidates in evaluation order; the miner
    driver evaluates each, records provenance, and either feeds the
    fitness back (for evolutionary search) or just consumes the
    iterator to completion (for random search).
    """

    def iterate(
        self,
        grammar: AlphaGrammar,
        rng: random.Random,
        fitness_fn: Callable[[Expression], float],
    ) -> Iterator[SearchYield]:
        """Yield ``(expression, generation, parent_name, mutation_kind)``."""
        ...


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomSearch:
    """Independent random samples from the grammar.

    Useful as a baseline and as the seed-population generator for
    :class:`EvolutionarySearch`. ``fitness_fn`` is accepted by the
    Protocol but not used here — random search doesn't condition on
    feedback.
    """

    n_candidates: int

    def __post_init__(self) -> None:
        if self.n_candidates <= 0:
            raise ValueError("RandomSearch.n_candidates must be > 0")

    def iterate(
        self,
        grammar: AlphaGrammar,
        rng: random.Random,
        fitness_fn: Callable[[Expression], float],
    ) -> Iterator[SearchYield]:
        del fitness_fn  # random search is feedback-free
        for _ in range(self.n_candidates):
            yield grammar.sample(rng), 0, None, None


# ---------------------------------------------------------------------------
# Evolutionary search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvolutionarySearch:
    """Tournament-selection genetic programming with mutation only.

    Generation 0 is a random seed population of size
    ``population_size``. Generations 1..N produce a fresh population
    of the same size by:

    1. **Elitism.** The top ``elite_size`` survivors of the previous
       generation are copied unchanged.
    2. **Tournament + mutate.** Until the new population is full,
       sample ``tournament_size`` candidates from the previous
       generation uniformly, take the highest-fitness one (winner),
       and append ``mutate(winner)`` to the new population.

    Parameters
    ----------
    population_size:
        Number of candidates per generation. Practical sweet spot is
        50–200 for typical search runs.
    n_generations:
        Number of generations to run beyond the seed. Total candidates
        evaluated is ``population_size * (n_generations + 1)``.
    elite_size:
        Top-K survivors copied unchanged each generation. Strict
        elitism (``elite_size >= 1``) is enough to guarantee best-
        fitness monotonicity.
    tournament_size:
        Number of candidates compared in each tournament. Larger →
        more aggressive selection pressure → faster convergence but
        higher risk of premature plateau.
    name_prefix:
        Prefix for the per-generation candidate names. The driver
        composes the full name as ``f"{name_prefix}{counter:06d}"``.
    """

    population_size: int = 32
    n_generations: int = 4
    elite_size: int = 2
    tournament_size: int = 3
    name_prefix: str = "gen"

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be > 0")
        if self.n_generations < 0:
            raise ValueError("n_generations must be >= 0")
        if self.elite_size < 0 or self.elite_size > self.population_size:
            raise ValueError("elite_size must lie in [0, population_size]")
        if self.tournament_size < 1 or self.tournament_size > self.population_size:
            raise ValueError("tournament_size must lie in [1, population_size]")

    def iterate(
        self,
        grammar: AlphaGrammar,
        rng: random.Random,
        fitness_fn: Callable[[Expression], float],
    ) -> Iterator[SearchYield]:
        """Yield candidates in evaluation order across all generations."""
        # Generation 0: random seeds. Track per-generation
        # (name, expression, fitness) tuples so the next generation
        # can run tournaments.
        seed_population: list[_PopulationEntry] = []
        for index in range(self.population_size):
            expr = grammar.sample(rng)
            name = self._candidate_name(0, index)
            yield expr, 0, None, None
            seed_population.append(
                _PopulationEntry(name=name, expression=expr, fitness=fitness_fn(expr))
            )

        prev_population = seed_population

        for generation in range(1, self.n_generations + 1):
            new_population: list[_PopulationEntry] = []
            # Elitism: sort by fitness desc, copy the top K.
            ranked = sorted(prev_population, key=lambda e: e.fitness, reverse=True)
            elites = ranked[: self.elite_size]
            for elite_index, elite in enumerate(elites):
                yield elite.expression, generation, elite.name, "elite"
                new_population.append(
                    _PopulationEntry(
                        name=self._candidate_name(generation, elite_index),
                        expression=elite.expression,
                        fitness=elite.fitness,
                    )
                )
            # Fill the remainder with tournament-selected mutants.
            index = self.elite_size
            while len(new_population) < self.population_size:
                competitors = rng.sample(prev_population, self.tournament_size)
                winner = max(competitors, key=lambda e: e.fitness)
                mutant, mutation_kind = mutate(winner.expression, grammar, rng)
                # Skip lookback-exploding mutants; otherwise the gate
                # rejects them and we burn the generation.
                if mutant.lookback_days() > grammar.max_total_lookback:
                    continue
                yield mutant, generation, winner.name, mutation_kind
                new_population.append(
                    _PopulationEntry(
                        name=self._candidate_name(generation, index),
                        expression=mutant,
                        fitness=fitness_fn(mutant),
                    )
                )
                index += 1

            prev_population = new_population

    def _candidate_name(self, generation: int, index: int) -> str:
        return f"{self.name_prefix}_{generation:02d}_{index:04d}"


@dataclass(frozen=True)
class _PopulationEntry:
    """Internal per-candidate record used during evolutionary iteration."""

    name: str
    expression: Expression
    fitness: float


# Re-exported so the runtime-checkable Protocol can be referred to
# from external code (e.g. user-defined searchers).
__all__ = [
    "EvolutionarySearch",
    "RandomSearch",
    "SearchAlgorithm",
    "SearchYield",
]
