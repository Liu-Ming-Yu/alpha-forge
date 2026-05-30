"""Pluggable alpha-model protocols for the walk-forward driver.

The walk-forward evaluator (:mod:`campaigns.evaluation.walk_forward`) refits a
model on each fold's training window and scores that fold's test samples.
Historically the only "model" was the linear IC-weighted ranker, inlined in the
driver. This module factors that boundary into two small protocols so other
learners (gradient-boosted trees first, sequence models later) can be swapped in
*without touching* the leakage discipline, eligibility gate, portfolio
constructor, or fold-streak dial.

Design: a model is a **factory** (:meth:`AlphaModel.fit`) that produces an
**immutable fitted object** (:class:`FittedAlphaModel`). Each fold gets a fresh
fit, so a fitted object never carries state across folds — the cleanest way to
keep the point-in-time contract impossible to violate by accident. ``fit`` sees
only the (already purged) training rows; ``score`` is called on test (and train,
for the volatility scale) rows of the same fold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


@runtime_checkable
class FittedAlphaModel(Protocol):
    """A model fitted to one fold's training window, ready to score samples."""

    def score(self, samples: Sequence[SupervisedAlphaSample]) -> list[float]:
        """Return one score per sample, in the same order as ``samples``.

        Higher score = more attractive. The driver ranks the test universe by
        this score; downstream IC and portfolio construction are computed from
        the ranking, so only the *relative* order across the cross-section
        matters (not the absolute scale).
        """
        ...

    def feature_weights(self) -> Mapping[str, float]:
        """Per-feature contribution used for evidence + feature-stability.

        For a linear model these are the actual coefficients; for a tree model
        they are normalized importances — a **reporting proxy** that does NOT
        drive :meth:`score`. The driver records this per fold and computes
        cross-fold ``feature_stability`` from it, so the only contract is that
        the values describe relative feature influence consistently across the
        folds of a single model.
        """
        ...


@runtime_checkable
class AlphaModel(Protocol):
    """A refittable alpha model: ``fit`` a fold's train window -> fitted model."""

    #: Stable identifier stamped into ``WalkForwardEvidence.model_version`` so an
    #: auditor can tell which learner produced an arm's evidence. Must not encode
    #: hardware (GPU vs CPU) — reruns on different machines should compare.
    name: str

    def fit(
        self,
        train: Sequence[SupervisedAlphaSample],
        feature_names: Sequence[str] | None = None,
    ) -> FittedAlphaModel:
        """Fit on this fold's training samples; return an immutable fitted model.

        ``feature_names`` restricts and orders the feature columns the model
        consumes; ``None`` means "every feature present in ``train``" (sorted
        for determinism). Implementations must treat a feature absent from a
        given sample's ``features`` dict as ``0.0`` — the same convention as the
        legacy ``score_features`` path.
        """
        ...
