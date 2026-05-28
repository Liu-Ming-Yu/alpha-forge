"""Auto-discovered formulaic alpha mining (Phase 4 of the brief).

The mining subsystem turns the formulaic engine + AST + operator
catalog into an *alpha factory*: feed it a panel of bars and a
forward-return label and it will generate, evaluate, and gate
thousands of candidate alphas without human handholding. Every
candidate the miner sees lands in an :class:`AutoAlphaProvenance`
record carrying full lineage (seed, generation, parent, mutation
kind), the evaluated evidence (IC, ICIR, turnover, correlation), and
the admission decision plus its reason — so the brief's "never store
only the name" requirement is satisfied by construction.

Public entry point: :func:`mine_alphas`. Public building blocks
re-exported here so callers can compose custom pipelines:

* :class:`AlphaGrammar` — what the search is allowed to sample from.
* :class:`RandomSearch`, :class:`EvolutionarySearch` — search loops.
* :class:`AdmissionGate` + :class:`AdmissionThresholds` — promotion
  gate with correlation pruning.
* :class:`CandidateEvidence` — per-candidate metrics record.
* :class:`AutoAlphaProvenance` — full provenance record.
* :func:`make_forward_return_labels` — convenience for deriving
  labels from the panel.
* :mod:`.mutation` — five AST mutators.

Two evaluation modes
--------------------

* **Single-pass** (default): one IC over the whole panel. Fast,
  convenient, biased — the same dates evaluate *and* admit a
  candidate. Use for development loops and quick smoke runs.
* **Walk-forward**: pass ``fold_config=MiningFoldConfig(...)`` to
  ``mine_alphas`` and every candidate gets K-fold OOS evidence
  (:class:`WalkForwardEvidence`). Per-fold ICs surface regime
  fragility via ``fold_negative_ic_streak`` and ``oos_icir``;
  :class:`AdmissionGate` enforces both via the
  ``max_fold_negative_ic_streak`` and ``min_n_folds_valid``
  thresholds without any other change at the call site.

Out of scope for this PR
-----------------------

* **RL-style search.** The Protocol shape is here so a future
  ``PolicySearch`` can drop in alongside random / evolutionary.
* **CLI operator workflow.** ``mine_alphas`` is a library function;
  a ``quant-platform mine-alphas`` CLI is a follow-up.
* **Auto-promotion of admitted alphas to the formulaic family's
  MANIFEST.** Admitted candidates today are returned for human
  review; auto-promotion lands once mining proves itself on real
  data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from random import Random
from typing import TYPE_CHECKING

from quant_platform.research.features.formulaic.config import (
    DEFAULT_CONFIG,
    FEATURE_SET_VERSION,
    OPERATOR_SET_VERSION,
)
from quant_platform.research.features.formulaic.evaluator import (
    ExpressionCache,
    evaluate_expression,
)
from quant_platform.research.features.formulaic.mining.admission import (
    AdmissionDecision,
    AdmissionGate,
    AdmissionThresholds,
)
from quant_platform.research.features.formulaic.mining.evidence import (
    CandidateEvidence,
    compute_evidence,
    make_forward_return_labels,
)
from quant_platform.research.features.formulaic.mining.grammar import (
    DEFAULT_LEAF_VARS,
    DEFAULT_POWERS,
    DEFAULT_WINDOWS,
    AlphaGrammar,
)
from quant_platform.research.features.formulaic.mining.mutation import (
    MUTATION_KINDS,
    mutate,
)
from quant_platform.research.features.formulaic.mining.provenance import (
    AutoAlphaProvenance,
)
from quant_platform.research.features.formulaic.mining.search import (
    EvolutionarySearch,
    RandomSearch,
    SearchAlgorithm,
)
from quant_platform.research.features.formulaic.mining.walk_forward import (
    MiningFoldConfig,
    WalkForwardEvidence,
    compute_walk_forward_evidence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from quant_platform.research.features.formulaic.ast import Expression
    from quant_platform.research.features.formulaic.panel import MarketPanel


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MiningResult:
    """Outcome of one :func:`mine_alphas` run.

    Attributes
    ----------
    admitted:
        Provenance records the gate admitted, in admission order.
    history:
        Every provenance record the miner produced, admitted or not.
        Use ``[p for p in history if not p.admitted]`` to inspect
        rejections.
    n_evaluated:
        ``len(history)``. Cached for the common case where callers
        only want the count.
    seed:
        Seed the miner ran with, copied out for logging.
    """

    admitted: tuple[AutoAlphaProvenance, ...]
    history: tuple[AutoAlphaProvenance, ...]
    n_evaluated: int
    seed: int


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def mine_alphas(
    *,
    grammar: AlphaGrammar,
    panel: MarketPanel,
    labels: pd.Series,
    search: SearchAlgorithm,
    gate: AdmissionGate,
    seed: int,
    operator_set_version: str = OPERATOR_SET_VERSION,
    feature_set_version: str = FEATURE_SET_VERSION,
    baseline_features: Mapping[str, pd.Series] | None = None,
    name_prefix: str = "auto_alpha",
    fold_config: MiningFoldConfig | None = None,
) -> MiningResult:
    """Run an alpha-mining pass.

    Parameters
    ----------
    grammar:
        Sampling vocabulary the search algorithm draws from.
    panel:
        :class:`MarketPanel` to score every candidate against.
    labels:
        Forward-return label Series, index-aligned to ``panel.frame``.
        :func:`make_forward_return_labels` is the supplied helper.
    search:
        :class:`SearchAlgorithm` instance — :class:`RandomSearch` or
        :class:`EvolutionarySearch` today; future
        ``PolicySearch`` etc. drops in via the Protocol.
    gate:
        :class:`AdmissionGate` deciding which evaluations get admitted.
    seed:
        RNG seed. Reproducibility contract: ``mine_alphas`` with
        identical inputs and the same ``seed`` produces identical
        history and admitted lists.
    operator_set_version:
        Operator catalog version pin stored on every provenance row.
        Defaults to :data:`OPERATOR_SET_VERSION`.
    feature_set_version:
        Feature-set version pin. Defaults to
        :data:`FEATURE_SET_VERSION`.
    baseline_features:
        Optional ``{name: Series}`` map of pre-existing features the
        miner should treat as the starting baseline for correlation
        pruning. Each admitted candidate is appended to this map as
        the run progresses, so later candidates see a growing
        baseline.
    name_prefix:
        Naming convention for generated candidates — the driver names
        the *i*-th evaluation ``f"{name_prefix}_{i:06d}"``. Default
        matches the brief's ``auto_alpha_001`` suggestion.
    fold_config:
        Optional :class:`MiningFoldConfig`. When provided, every
        candidate is scored with K-fold OOS evidence via
        :func:`compute_walk_forward_evidence` instead of the single-
        pass :func:`compute_evidence`. The resulting
        :class:`WalkForwardEvidence` lands on
        :attr:`AutoAlphaProvenance.evidence` and the gate's
        WF-specific thresholds
        (``max_fold_negative_ic_streak``, ``min_n_folds_valid``)
        fire automatically.

    Returns
    -------
    MiningResult
    """
    # ``Random`` is deterministic for reproducibility, not for crypto;
    # the lint warning is irrelevant to alpha mining.
    rng = Random(seed)  # noqa: S311
    expression_cache = ExpressionCache()

    # Mutable per-run baseline. Starts with the caller-supplied
    # baselines (e.g. the curated formulaic library) and grows as
    # admissions happen — diversity pressure compounds.
    running_baseline: dict[str, pd.Series] = dict(baseline_features or {})
    evidence_cache: dict[Expression, CandidateEvidence | WalkForwardEvidence] = {}

    admitted: list[AutoAlphaProvenance] = []
    history: list[AutoAlphaProvenance] = []

    def _fitness(expr: Expression) -> float:
        # See module docstring: the search loop calls ``fitness_fn``
        # only after the driver has populated ``evidence_cache`` for
        # that expression. Missing entries are a programmer error;
        # we return ``-inf`` so the candidate loses every tournament
        # rather than crash mid-search.
        cached = evidence_cache.get(expr)
        if cached is None:
            return float("-inf")
        # Use ``-inf`` instead of NaN so the comparison behaviour in
        # ``max(...)`` is well-defined even when an alpha's evidence
        # came back all-NaN (lookback-blown-out or constant feature).
        return cached.rank_ic if cached.rank_ic == cached.rank_ic else float("-inf")

    for index, (expression, generation, parent_name, mutation_kind) in enumerate(
        search.iterate(grammar, rng, _fitness),
        start=1,
    ):
        # Avoid re-evaluating the same expression. Elites get yielded
        # again in :class:`EvolutionarySearch`; identical mutants can
        # also recur. ``Expression`` is a frozen dataclass with
        # structural hash, so the dict-key check is exact.
        if expression in evidence_cache:
            evidence = evidence_cache[expression]
        elif fold_config is not None:
            evidence = compute_walk_forward_evidence(
                expression,
                panel,
                labels,
                fold_config=fold_config,
                baseline_features=running_baseline,
                cache=expression_cache,
            )
            evidence_cache[expression] = evidence
        else:
            evidence = compute_evidence(
                expression,
                panel,
                labels,
                baseline_features=running_baseline,
                cache=expression_cache,
            )
            evidence_cache[expression] = evidence

        candidate_name = f"{name_prefix}_{index:06d}"
        prov = AutoAlphaProvenance(
            name=candidate_name,
            expression=expression,
            generation=generation,
            seed=seed,
            parent_name=parent_name,
            mutation_kind=mutation_kind,
            operator_set_version=operator_set_version,
            feature_set_version=feature_set_version,
            evidence=evidence,
            created_at=datetime.now(UTC),
            admitted=False,
            admission_reason="",
        )

        decision = gate.evaluate(prov, admitted, panel_size=len(panel.frame))
        recorded = _finalise_provenance(prov, decision)
        history.append(recorded)
        if decision.admitted:
            admitted.append(recorded)
            # Update the running baseline so the next candidate's
            # correlation-to-admitted check sees this row.
            running_baseline[candidate_name] = evaluate_expression(
                panel, expression, cache=expression_cache
            )

    return MiningResult(
        admitted=tuple(admitted),
        history=tuple(history),
        n_evaluated=len(history),
        seed=seed,
    )


def _finalise_provenance(
    provenance: AutoAlphaProvenance,
    decision: AdmissionDecision,
) -> AutoAlphaProvenance:
    """Stamp the admission decision onto a provenance record.

    The provenance dataclass is frozen, so this allocates a new
    instance — kept in a helper so the call site reads as one verb.
    """
    from dataclasses import replace

    return replace(
        provenance,
        admitted=decision.admitted,
        admission_reason=decision.reason,
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_LEAF_VARS",
    "DEFAULT_POWERS",
    "DEFAULT_WINDOWS",
    "FEATURE_SET_VERSION",
    "MUTATION_KINDS",
    "OPERATOR_SET_VERSION",
    "AdmissionDecision",
    "AdmissionGate",
    "AdmissionThresholds",
    "AlphaGrammar",
    "AutoAlphaProvenance",
    "CandidateEvidence",
    "EvolutionarySearch",
    "MiningFoldConfig",
    "MiningResult",
    "RandomSearch",
    "SearchAlgorithm",
    "WalkForwardEvidence",
    "compute_evidence",
    "compute_walk_forward_evidence",
    "make_forward_return_labels",
    "mine_alphas",
    "mutate",
]
