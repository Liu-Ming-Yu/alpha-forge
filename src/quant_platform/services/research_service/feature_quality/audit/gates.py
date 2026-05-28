"""Runner-facing feature-audit gate adapter methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.feature_quality.audit.gate_evaluators import (
    evaluate_cost_gate,
    evaluate_economic_gate,
    evaluate_incremental_gate,
    evaluate_leakage_gate,
    evaluate_noise_gate,
    evaluate_stability_gate,
)
from quant_platform.services.research_service.feature_quality.audit.thresholds import (
    FeatureAuditThresholds,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.core.domain.research import FeatureDefinition
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

__all__ = ["FeatureAuditGatesMixin", "FeatureAuditThresholds"]


class FeatureAuditGatesMixin:
    """Six feature-admission gates used by FeatureAuditRunner."""

    _thresholds: FeatureAuditThresholds
    _rng_seed: int
    _slippage_bps: float
    _baseline_features: tuple[str, ...]

    def _noise_gate(
        self,
        samples: Sequence[SupervisedAlphaSample],
        feature_name: str,
    ) -> dict[str, object]:
        return evaluate_noise_gate(samples, feature_name, self._thresholds)

    def _leakage_gate(
        self,
        feature: FeatureDefinition,
        samples: Sequence[SupervisedAlphaSample],
        rows: Sequence[SupervisedAlphaSample],
    ) -> dict[str, object]:
        return evaluate_leakage_gate(feature, samples, rows, self._thresholds, self._rng_seed)

    def _stability_gate(
        self,
        feature: FeatureDefinition,
        rows: Sequence[SupervisedAlphaSample],
    ) -> dict[str, object]:
        return evaluate_stability_gate(feature, rows, self._thresholds, self._rng_seed)

    def _economic_gate(
        self,
        feature: FeatureDefinition,
        stability: Mapping[str, object],
    ) -> dict[str, object]:
        return evaluate_economic_gate(feature, stability)

    def _cost_gate(
        self,
        feature: FeatureDefinition,
        rows: Sequence[SupervisedAlphaSample],
    ) -> dict[str, object]:
        return evaluate_cost_gate(feature, rows, self._thresholds, self._slippage_bps)

    def _incremental_gate(
        self,
        feature: FeatureDefinition,
        samples: Sequence[SupervisedAlphaSample],
    ) -> dict[str, object]:
        return evaluate_incremental_gate(
            feature,
            samples,
            self._thresholds,
            self._baseline_features,
        )
