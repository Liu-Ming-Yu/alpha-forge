"""Linear IC-weighted ranker as an :class:`AlphaModel`.

This is the platform's original ranker — per-feature Spearman IC, normalized to
weights, scored as a weighted sum over **cross-sectionally rank-normalized**
features. The fit is unchanged from the legacy ``fit_correlation_weights``
(rank-based Spearman IC), but scoring no longer takes a *raw* weighted sum:
raw features span ~8 orders of magnitude (``dollar_volume_20d`` is raw dollars
~1e8 vs returns ~[-1, 1]), so a raw ``Σ feature·weight`` collapsed the model
into a single-feature (dollar-volume) sort. Scoring now rank-normalizes each
feature within its as-of cross-section first (see :class:`_FittedLinearRanker`),
which leaves the fitted weights untouched (Spearman is invariant to the
monotonic transform) while making each feature contribute in proportion to its
weight. The live scorer applies the identical normalization in the feature
bundle, so live and backtest scores agree by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    equal_weights,
    fit_correlation_weights,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    cross_sectional_rank_normalize,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

WeightMode = Literal["ic_weighted", "equal_weight"]

#: Per-call date column used to group a scoring batch into per-date
#: cross-sections for rank-normalization. Prefixed to avoid colliding
#: with any real feature name.
_RANK_DATE_COLUMN = "__rank_as_of"


class _FittedLinearRanker:
    """Immutable per-fold fit: a frozen weight vector + weighted-sum scoring.

    Scoring rank-normalizes each weighted feature **within its as-of
    cross-section** before the weighted sum. Raw features span ~8 orders of
    magnitude (e.g. ``dollar_volume_20d`` is raw dollars ~1e8 vs returns
    ~[-1, 1]); a raw ``Σ feature·weight`` is therefore dominated by the
    largest-scale feature regardless of its IC weight, collapsing the
    multi-factor model into a single-feature sort. Rank-normalization
    (:func:`cross_sectional_rank_normalize`) puts every feature on a common
    per-date scale so each contributes in proportion to its weight. The IC
    weights are fit by rank-based Spearman correlation (see
    :func:`fit_correlation_weights`), which is invariant to this monotonic
    transform — so the fitted weights are unchanged; only the scoring is
    corrected. The live scorer applies the same normalization in the feature
    bundle, so live and backtest scores agree by construction.
    """

    __slots__ = ("_weights",)

    def __init__(self, weights: Mapping[str, float]) -> None:
        self._weights = dict(weights)

    def score(self, samples: Sequence[SupervisedAlphaSample]) -> list[float]:
        if not samples:
            return []
        names = list(self._weights)
        frame = pd.DataFrame(
            {
                _RANK_DATE_COLUMN: [row.as_of for row in samples],
                **{
                    name: [float(row.features.get(name, float("nan"))) for row in samples]
                    for name in names
                },
            }
        )
        normed = cross_sectional_rank_normalize(frame, names, date_column=_RANK_DATE_COLUMN)
        # Σ rank-normalized feature · weight (identical to score_features on
        # the normalized features; vectorised over the batch for speed).
        weighted = normed[names].mul(pd.Series(self._weights)).sum(axis=1)
        return [float(value) for value in weighted]

    def feature_weights(self) -> Mapping[str, float]:
        return dict(self._weights)


class LinearICRanker:
    """IC-weighted (or equal-weight) linear ranker.

    ``non_negative`` drops factors with negative in-sample IC rather than
    shorting them — the platform's classical factors are positive-oriented, so a
    negative weight is regime-overfit (it shorts a factor with a real positive
    long-term premium). This mirrors the driver's prior default of
    ``fit_correlation_weights(..., non_negative=True)``; the default ``name``
    therefore matches the legacy ``MODEL_VERSION``. Note the fitted *weights* are
    unchanged from the legacy path, but *scoring* now rank-normalizes features
    (see module docstring), so linear-arm evidence changes versus pre-fix runs.
    """

    def __init__(
        self,
        *,
        weight_mode: WeightMode = "ic_weighted",
        non_negative: bool = True,
    ) -> None:
        if weight_mode not in ("ic_weighted", "equal_weight"):
            raise ValueError(f"unsupported weight_mode: {weight_mode!r}")
        self._weight_mode: WeightMode = weight_mode
        self._non_negative = non_negative
        if weight_mode == "ic_weighted":
            self.name = "ic-weighted-non-negative" if non_negative else "ic-weighted"
        else:
            self.name = "equal-weight"

    def fit(
        self,
        train: Sequence[SupervisedAlphaSample],
        feature_names: Sequence[str] | None = None,
    ) -> _FittedLinearRanker:
        if self._weight_mode == "equal_weight":
            weights = equal_weights(train, feature_names)
        else:
            weights = fit_correlation_weights(
                train,
                feature_names,
                non_negative=self._non_negative,
            )
        return _FittedLinearRanker(weights)
