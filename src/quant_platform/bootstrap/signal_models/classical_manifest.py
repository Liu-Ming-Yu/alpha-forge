"""Load campaign-fitted classical signal weights from a research manifest.

A research campaign (``research-campaign run``) fits the classical factor
weights from data via walk-forward IC and writes them to a campaign manifest
(``campaign_manifest.json``) as ``selected_weights``.  This module is the
missing link that loads those data-driven weights into the runtime model,
replacing the hand-picked ``FactorSettings`` defaults.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from quant_platform.services.signal_service.scoring import LinearWeightSignalModel

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger(__name__)


def load_classical_signal_model(manifest_path: Path) -> LinearWeightSignalModel:
    """Build a ``LinearWeightSignalModel`` from a campaign manifest.

    The model is pinned to the manifest's ``feature_set_version`` so a stale or
    mismatched feature rebuild fails closed at scoring time.  Weights are loaded
    regardless of campaign eligibility — promotion is governed separately by the
    production-candidate gate — but an ineligible manifest is logged loudly.

    Raises:
        FileNotFoundError: the manifest path does not exist.
        ValueError: the manifest is malformed or carries no usable weights.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Classical weights manifest not found: {manifest_path}. "
            "Set QP__FACTORS__FITTED_WEIGHTS_MANIFEST to a campaign manifest, "
            "or clear it to use the hand-picked factor defaults."
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed classical weights manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Classical weights manifest {manifest_path} is not a JSON object.")

    weights = _extract_weights(payload, manifest_path)
    feature_set_version = payload.get("feature_set_version")
    model_version = str(payload.get("model_version") or "campaign-fitted")
    passed = bool(payload.get("passed", False))

    if not passed:
        log.warning(
            "classical_manifest.not_promotion_eligible",
            manifest=str(manifest_path),
            model_version=model_version,
            hint="campaign eligibility 'passed' is false; weights loaded for research only",
        )
    log.info(
        "classical_manifest.loaded",
        manifest=str(manifest_path),
        model_version=model_version,
        feature_set_version=feature_set_version,
        n_weights=len(weights),
        passed=passed,
    )
    return LinearWeightSignalModel(
        weights,
        model_version=model_version,
        expected_feature_set_version=(str(feature_set_version) if feature_set_version else None),
    )


def _extract_weights(payload: dict[str, object], manifest_path: Path) -> dict[str, float]:
    """Pull non-zero numeric ``selected_weights`` from a campaign manifest."""
    raw_weights = payload.get("selected_weights")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError(f"Classical weights manifest {manifest_path} has no 'selected_weights'.")
    weights: dict[str, float] = {}
    for name, value in raw_weights.items():
        try:
            weight = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Non-numeric weight for {name!r} in {manifest_path}: {value!r}"
            ) from exc
        if weight != 0.0:
            weights[str(name)] = weight
    if not weights:
        raise ValueError(f"Classical weights manifest {manifest_path} has only zero weights.")
    return weights


__all__ = ["load_classical_signal_model"]
