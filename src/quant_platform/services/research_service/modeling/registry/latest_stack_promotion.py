"""Adapter: latest-stack backtest evidence → model-registry registration.

ADR-004 Action Item 14. The latest-stack backtest emits per-arm evidence JSON
(schema ``backtest-latest-stack-realized-v2.x``) carrying the eligibility-gate
result and headline metrics. This adapter turns an *eligible* production-
candidate arm's evidence into the inputs the model registry consumes —
``register_model(strategy_name, model_version, feature_set_version, as_of,
metadata)`` — so a winning arm (e.g. Arm G) enters the promotion sequence
without an operator hand-assembling the call.

It is a pure metadata/evidence conversion: the registry stores no weights
(linear-ranker weights are recomputed at live feature time; trained model
artifacts are referenced through the governance ``alpha_promote`` path, which is
the live, DSN-backed step this adapter feeds).

Promotion is gated on the backtest's own verdict — the adapter never re-derives
thresholds. Only an arm whose evidence shows ``eligibility.passed`` AND is a
tagged ``portfolio_candidate`` AND is flagged ``production_candidate`` may be
registered; diagnostic baselines are rejected loudly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_platform.core.contracts.model_registry import (
        ModelRegistryRepository,
        RegisteredModelRecord,
    )

#: Metrics copied into the registry metadata so the promotion decision is
#: auditable from the registry alone, without re-opening the backtest evidence.
_HEADLINE_METRIC_KEYS = (
    "slippage_adjusted_sharpe",
    "max_drawdown",
    "fold_negative_ic_streak",
    "max_drawdown_during_worst_streak",
    "oos_rolling_ic",
    "ic_60d",
    "total_return",
)


class NotPromotableError(ValueError):
    """Raised when an arm's evidence does not qualify for registry promotion."""


@dataclass(frozen=True)
class ModelRegistration:
    """Validated inputs for ``ModelRegistryRepository.register_model``."""

    strategy_name: str
    model_version: str
    feature_set_version: str
    as_of: datetime
    metadata: dict[str, object] = field(default_factory=dict)


def build_registration(
    evidence: Mapping[str, object],
    *,
    as_of: datetime | None = None,
) -> ModelRegistration:
    """Build a registry registration from one latest-stack arm's evidence.

    Raises :class:`NotPromotableError` if the arm is not a ``portfolio_candidate``,
    is not flagged ``production_candidate``, or did not pass its eligibility
    gate. ``as_of`` defaults to the evidence's ``saved_at_utc`` timestamp.
    """
    category = evidence.get("arm_category")
    if category != "portfolio_candidate":
        raise NotPromotableError(f"only portfolio_candidate arms are promotable; got {category!r}")
    if not evidence.get("production_candidate"):
        raise NotPromotableError("arm is not flagged production_candidate")
    eligibility = evidence.get("eligibility")
    if not isinstance(eligibility, Mapping) or not eligibility.get("passed"):
        raise NotPromotableError("arm did not pass its eligibility gate")

    strategy_name = _required_str(evidence, "arm")
    model_version = _required_str(evidence, "model_version")
    feature_set_version = _required_str(evidence, "feature_set_version")
    resolved_as_of = as_of if as_of is not None else _parse_saved_at(evidence)

    metrics = evidence.get("metrics")
    metrics_map = metrics if isinstance(metrics, Mapping) else {}
    metadata: dict[str, object] = {
        "source": "backtest_latest_stack",
        "evidence_schema_version": evidence.get("evidence_schema_version"),
        "run_id": evidence.get("run_id"),
        "arm_cli_alias": evidence.get("arm_cli_alias"),
        "arm_category": category,
        "git_commit": evidence.get("git_commit"),
        # The full gate result is the promotion justification.
        "eligibility": dict(eligibility),
        "eligibility_thresholds": _as_dict(evidence.get("eligibility_thresholds")),
        "headline_metrics": {k: metrics_map[k] for k in _HEADLINE_METRIC_KEYS if k in metrics_map},
        "universe_fingerprint": _as_dict(evidence.get("universe_fingerprint")),
        "bars_snapshot_fingerprint": _as_dict(evidence.get("bars_snapshot_fingerprint")),
    }
    return ModelRegistration(
        strategy_name=strategy_name,
        model_version=model_version,
        feature_set_version=feature_set_version,
        as_of=resolved_as_of,
        metadata=metadata,
    )


async def promote_to_registry(
    registry: ModelRegistryRepository,
    evidence: Mapping[str, object],
    *,
    as_of: datetime | None = None,
) -> RegisteredModelRecord:
    """Build a registration from ``evidence`` and register it.

    Thin wrapper over :func:`build_registration` + ``register_model``. Raises
    :class:`NotPromotableError` (before touching the registry) if the arm does not
    qualify, so a rejected arm never reaches persistence.
    """
    registration = build_registration(evidence, as_of=as_of)
    return await registry.register_model(
        strategy_name=registration.strategy_name,
        model_version=registration.model_version,
        feature_set_version=registration.feature_set_version,
        as_of=registration.as_of,
        metadata=registration.metadata,
    )


def _required_str(evidence: Mapping[str, object], key: str) -> str:
    value = evidence.get(key)
    if not isinstance(value, str) or not value:
        raise NotPromotableError(f"evidence missing required string field {key!r}")
    return value


def _parse_saved_at(evidence: Mapping[str, object]) -> datetime:
    raw = evidence.get("saved_at_utc")
    if not isinstance(raw, str):
        raise NotPromotableError("evidence has no saved_at_utc timestamp; pass as_of explicitly")
    return datetime.fromisoformat(raw)


def _as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "ModelRegistration",
    "NotPromotableError",
    "build_registration",
    "promote_to_registry",
]
