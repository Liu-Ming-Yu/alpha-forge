"""Signal scoring engine: feature vectors → signal scores.

This module provides a pluggable scoring interface and a simple
linear-weight reference implementation.  In production, the scoring
function is replaced by a trained model.

Research-to-production parity:
    The SignalModel protocol is shared between backtest and live paths.
    Only the features and timing differ, not the scoring logic.
"""

from __future__ import annotations

import math
import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.signals import SignalScore
from quant_platform.core.exceptions import DataStalenessError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.core.domain.research import FeatureVector, StrategyRun

log = structlog.get_logger(__name__)


class LinearWeightSignalModel:
    """Reference implementation: weighted sum of features, normalised to [-1, 1].

    Args:
        feature_weights: Mapping of feature_name → weight.
            Features not in this mapping are ignored.
        model_version: Semantic version string for this model.
        strict_missing: When True, raise DataStalenessError if any weighted
            feature is absent from a vector instead of silently using 0.0.
        expected_feature_set_version: When set, validate that every incoming
            FeatureVector carries this exact feature_set_version.  None = disabled.
    """

    def __init__(
        self,
        feature_weights: Mapping[str, float],
        model_version: str = "0.1.0",
        strict_missing: bool = False,
        expected_feature_set_version: str | None = None,
    ) -> None:
        self._weights = dict(feature_weights)
        self._model_version = model_version
        self._strict_missing = strict_missing
        self._expected_fsv = expected_feature_set_version

    def score(
        self,
        vectors: list[FeatureVector],
        strategy_run: StrategyRun,
    ) -> list[SignalScore]:
        """Compute a signal score for each feature vector.

        Confidence reflects feature coverage: the fraction of weighted features
        present in the vector.  A fully-covered vector scores 1.0; a vector
        with half its features missing scores 0.5.
        """
        scores: list[SignalScore] = []
        n_weights = max(1, len(self._weights))
        # Even when the caller has not pinned an ``expected_feature_set_version``,
        # every vector in a single ``score()`` call must agree on its
        # feature_set_version: a mixed batch is symptom of a stale/partial
        # rebuild and would produce silently inconsistent scores.
        observed_fsv: str | None = None
        for vec in vectors:
            if self._expected_fsv is not None and vec.feature_set_version != self._expected_fsv:
                raise ValueError(
                    f"Feature set version mismatch for instrument {vec.instrument_id}: "
                    f"expected {self._expected_fsv!r}, got {vec.feature_set_version!r}"
                )
            if self._expected_fsv is None:
                if observed_fsv is None:
                    observed_fsv = vec.feature_set_version
                elif vec.feature_set_version != observed_fsv:
                    raise ValueError(
                        "Inconsistent feature_set_version within scoring batch: "
                        f"first vector reported {observed_fsv!r}, but vector for "
                        f"instrument {vec.instrument_id} reported "
                        f"{vec.feature_set_version!r}. Pin "
                        "expected_feature_set_version on the model to make this "
                        "intentional, or rebuild features to a single version."
                    )
            raw = 0.0
            covered = 0
            for name, weight in self._weights.items():
                if name not in vec.features:
                    if self._strict_missing:
                        raise DataStalenessError(
                            f"Required feature {name!r} missing from vector for "
                            f"instrument {vec.instrument_id}",
                            instrument_id=vec.instrument_id,
                        )
                    log.warning(
                        "signal.missing_feature",
                        feature=name,
                        instrument_id=str(vec.instrument_id),
                        model_version=self._model_version,
                    )
                    continue
                value = vec.features[name]
                if not math.isfinite(value):
                    raise DataStalenessError(
                        f"Non-finite value {value!r} for feature {name!r} on "
                        f"instrument {vec.instrument_id}",
                        instrument_id=vec.instrument_id,
                    )
                raw += value * weight
                covered += 1
            clamped = max(-1.0, min(1.0, raw))
            confidence = covered / n_weights
            scores.append(
                SignalScore(
                    score_id=uuid.uuid4(),
                    instrument_id=vec.instrument_id,
                    strategy_run_id=strategy_run.run_id,
                    as_of=vec.as_of,
                    score=clamped,
                    confidence=confidence,
                    model_version=self._model_version,
                    feature_vector_id=vec.vector_id,
                )
            )
        return scores

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self._weights)

    @property
    def expected_feature_set_version(self) -> str | None:
        return self._expected_fsv
