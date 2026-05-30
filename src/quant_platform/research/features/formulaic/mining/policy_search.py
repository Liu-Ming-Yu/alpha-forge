"""Policy-guided formulaic-alpha search.

This module applies the qlib RL shape to alpha mining:

``AlphaMutationSimulator`` owns the candidate-expression trajectory,
``AlphaSearchStateInterpreter`` turns that state into a compact observation,
``AlphaSearchActionInterpreter`` validates the policy action, and
``PolicySearch`` exposes the same ``SearchAlgorithm`` surface as random and
evolutionary search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from quant_platform.research.features.formulaic.mining.mutation import MUTATION_KINDS
from quant_platform.research.features.formulaic.mining.mutation_actions import mutate_with_kind
from quant_platform.research.rl import ActionInterpreter, Policy, StateInterpreter

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterator

    from quant_platform.research.features.formulaic.ast import Expression
    from quant_platform.research.features.formulaic.mining.grammar import AlphaGrammar


@dataclass(frozen=True)
class AlphaSearchState:
    """Simulator state for one formulaic-alpha mutation trajectory."""

    expression: Expression
    generation: int
    parent_name: str | None
    mutation_kind: str | None
    last_fitness: float | None
    best_fitness: float | None
    stagnation_steps: int
    max_total_lookback: int


@dataclass(frozen=True)
class AlphaSearchObservation:
    """Policy-facing summary of the current expression trajectory."""

    generation: int
    node_count: int
    lookback_days: int
    lookback_fraction: float
    stagnation_steps: int
    last_fitness: float | None
    best_fitness: float | None
    last_mutation_kind: str | None


@dataclass(frozen=True)
class AlphaSearchPolicyAction:
    """Raw action emitted by an alpha-search policy."""

    mutation_kind: str


@dataclass(frozen=True)
class AlphaMutationCommand:
    """Simulator-facing mutation command."""

    mutation_kind: str


class AlphaMutationSimulator:
    """Finite simulator over formulaic-alpha mutations."""

    def __init__(
        self,
        *,
        initial: Expression,
        grammar: AlphaGrammar,
        rng: random.Random,
        max_generation: int,
        max_mutation_attempts: int = 16,
        name_prefix: str = "policy",
    ) -> None:
        if max_generation < 0:
            raise ValueError("max_generation must be >= 0")
        if max_mutation_attempts <= 0:
            raise ValueError("max_mutation_attempts must be > 0")
        self._expression = initial
        self._grammar = grammar
        self._rng = rng
        self._max_generation = max_generation
        self._max_mutation_attempts = max_mutation_attempts
        self._name_prefix = name_prefix
        self._generation = 0
        self._parent_name: str | None = None
        self._mutation_kind: str | None = None
        self._last_fitness: float | None = None
        self._best_fitness: float | None = None
        self._stagnation_steps = 0

    def step(self, action: AlphaMutationCommand) -> None:
        """Apply a mutation chosen by the policy."""
        if self.done():
            raise RuntimeError("cannot step a completed alpha-search trajectory")
        parent_name = self._candidate_name(self._generation)
        self._expression = self._bounded_mutation(action.mutation_kind)
        self._generation += 1
        self._parent_name = parent_name
        self._mutation_kind = action.mutation_kind

    def record_fitness(self, fitness: float) -> None:
        """Feed externally-computed candidate fitness back into the state."""
        self._last_fitness = float(fitness)
        if self._best_fitness is None or fitness > self._best_fitness:
            self._best_fitness = float(fitness)
            self._stagnation_steps = 0
        else:
            self._stagnation_steps += 1

    def get_state(self) -> AlphaSearchState:
        return AlphaSearchState(
            expression=self._expression,
            generation=self._generation,
            parent_name=self._parent_name,
            mutation_kind=self._mutation_kind,
            last_fitness=self._last_fitness,
            best_fitness=self._best_fitness,
            stagnation_steps=self._stagnation_steps,
            max_total_lookback=self._grammar.max_total_lookback,
        )

    def done(self) -> bool:
        return self._generation >= self._max_generation

    def _bounded_mutation(self, kind: str) -> Expression:
        for _ in range(self._max_mutation_attempts):
            mutated, _ = mutate_with_kind(self._expression, self._grammar, self._rng, kind)
            if mutated.lookback_days() <= self._grammar.max_total_lookback:
                return mutated
        # No within-budget mutation after max_mutation_attempts: keep the
        # current (already-valid) expression unchanged. The step still advances
        # the generation, so PolicySearch always yields exactly n_candidates;
        # mine_alphas dedupes identical expressions via its evidence_cache, so a
        # no-op fallback costs at most one extra provenance row, never a second
        # evaluation. This mirrors EvolutionarySearch tolerating mutate()
        # returning its input unchanged when no valid mutation applies.
        return self._expression

    def _candidate_name(self, generation: int) -> str:
        return f"{self._name_prefix}_{generation:06d}"


class AlphaSearchStateInterpreter(StateInterpreter[AlphaSearchState, AlphaSearchObservation]):
    """Compact state summary for mutation policies."""

    def interpret(self, simulator_state: AlphaSearchState) -> AlphaSearchObservation:
        lookback_days = simulator_state.expression.lookback_days()
        max_lookback = max(1, simulator_state.max_total_lookback)
        return AlphaSearchObservation(
            generation=simulator_state.generation,
            node_count=sum(1 for _ in simulator_state.expression.walk()),
            lookback_days=lookback_days,
            lookback_fraction=lookback_days / max_lookback,
            stagnation_steps=simulator_state.stagnation_steps,
            last_fitness=simulator_state.last_fitness,
            best_fitness=simulator_state.best_fitness,
            last_mutation_kind=simulator_state.mutation_kind,
        )


class AlphaSearchActionInterpreter(
    ActionInterpreter[AlphaSearchState, AlphaSearchPolicyAction | str, AlphaMutationCommand]
):
    """Validate policy-selected mutation actions."""

    def interpret(
        self,
        simulator_state: AlphaSearchState,
        action: AlphaSearchPolicyAction | str,
    ) -> AlphaMutationCommand:
        del simulator_state
        kind = action if isinstance(action, str) else action.mutation_kind
        return AlphaMutationCommand(mutation_kind=kind)

    def validate_action(self, action: AlphaSearchPolicyAction | str) -> None:
        kind = action if isinstance(action, str) else action.mutation_kind
        if kind not in MUTATION_KINDS:
            valid = ", ".join(MUTATION_KINDS)
            raise ValueError(f"unknown mutation kind {kind!r}; valid kinds: {valid}")


@dataclass(frozen=True)
class FitnessAwareMutationPolicy:
    """Small deterministic policy for guided alpha search.

    It prunes expressions near the lookback budget, explores when recent
    mutations stagnate, and otherwise grows or swaps the current tree. The
    policy is deliberately simple; learned policies can replace it behind the
    same ``Policy`` contract.
    """

    growth_kind: str = "wrap_op"
    prune_lookback_fraction: float = 0.85
    stagnation_patience: int = 2
    exploration_kinds: tuple[str, ...] = ("swap_operator", "change_window", "replace_var")
    stagnation_kinds: tuple[str, ...] = ("replace_var", "swap_operator", "prune")

    def __post_init__(self) -> None:
        kinds = (self.growth_kind, *self.exploration_kinds, *self.stagnation_kinds, "prune")
        invalid = [kind for kind in kinds if kind not in MUTATION_KINDS]
        if invalid:
            raise ValueError(f"invalid mutation kinds: {tuple(invalid)!r}")
        if not 0.0 <= self.prune_lookback_fraction <= 1.0:
            raise ValueError("prune_lookback_fraction must be in [0, 1]")
        if self.stagnation_patience < 0:
            raise ValueError("stagnation_patience must be >= 0")

    def act(self, observation: AlphaSearchObservation) -> AlphaSearchPolicyAction:
        if observation.lookback_fraction >= self.prune_lookback_fraction:
            return AlphaSearchPolicyAction("prune")
        if observation.last_fitness is None:
            return AlphaSearchPolicyAction(self.growth_kind)
        if observation.stagnation_steps >= self.stagnation_patience:
            index = observation.generation % len(self.stagnation_kinds)
            return AlphaSearchPolicyAction(self.stagnation_kinds[index])
        if (
            observation.best_fitness is not None
            and observation.last_fitness >= observation.best_fitness
        ):
            return AlphaSearchPolicyAction(self.growth_kind)
        index = observation.generation % len(self.exploration_kinds)
        return AlphaSearchPolicyAction(self.exploration_kinds[index])


@dataclass(frozen=True)
class PolicySearch:
    """SearchAlgorithm implementation driven by a mutation policy."""

    n_candidates: int
    policy: Policy[AlphaSearchObservation, AlphaSearchPolicyAction | str] | None = None
    state_interpreter: AlphaSearchStateInterpreter = field(
        default_factory=AlphaSearchStateInterpreter
    )
    action_interpreter: AlphaSearchActionInterpreter = field(
        default_factory=AlphaSearchActionInterpreter
    )
    max_mutation_attempts: int = 16
    name_prefix: str = "policy"

    def __post_init__(self) -> None:
        if self.n_candidates <= 0:
            raise ValueError("PolicySearch.n_candidates must be > 0")
        if self.max_mutation_attempts <= 0:
            raise ValueError("max_mutation_attempts must be > 0")

    def iterate(
        self,
        grammar: AlphaGrammar,
        rng: random.Random,
        fitness_fn: Callable[[Expression], float],
    ) -> Iterator[tuple[Expression, int, str | None, str | None]]:
        policy = self.policy or FitnessAwareMutationPolicy()
        simulator = AlphaMutationSimulator(
            initial=grammar.sample(rng),
            grammar=grammar,
            rng=rng,
            max_generation=self.n_candidates - 1,
            max_mutation_attempts=self.max_mutation_attempts,
            name_prefix=self.name_prefix,
        )

        state = simulator.get_state()
        yield state.expression, state.generation, state.parent_name, state.mutation_kind
        simulator.record_fitness(fitness_fn(state.expression))

        while not simulator.done():
            state = simulator.get_state()
            observation = self.state_interpreter(state)
            policy_action = policy.act(observation)
            command = self.action_interpreter(state, policy_action)
            simulator.step(command)
            state = simulator.get_state()
            yield state.expression, state.generation, state.parent_name, state.mutation_kind
            simulator.record_fitness(fitness_fn(state.expression))


__all__ = [
    "AlphaMutationCommand",
    "AlphaMutationSimulator",
    "AlphaSearchActionInterpreter",
    "AlphaSearchObservation",
    "AlphaSearchPolicyAction",
    "AlphaSearchState",
    "AlphaSearchStateInterpreter",
    "FitnessAwareMutationPolicy",
    "PolicySearch",
]
