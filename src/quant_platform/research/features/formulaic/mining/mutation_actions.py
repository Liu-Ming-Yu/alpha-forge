"""Action-oriented access to formulaic-alpha mutation primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.research.features.formulaic.mining.mutation import mutator_for_kind

if TYPE_CHECKING:
    import random

    from quant_platform.research.features.formulaic.ast import Expression
    from quant_platform.research.features.formulaic.mining.grammar import AlphaGrammar


def mutate_with_kind(
    expr: Expression,
    grammar: AlphaGrammar,
    rng: random.Random,
    kind: str,
) -> tuple[Expression, str]:
    """Apply a specific mutation kind.

    Policy-guided search uses this to turn a policy action into the same
    primitive mutators that random and evolutionary search already use.
    Raises :class:`ValueError` (via :func:`mutator_for_kind`) for an
    unknown kind.
    """
    return mutator_for_kind(kind)(expr, grammar, rng), kind


__all__ = ["mutate_with_kind"]
