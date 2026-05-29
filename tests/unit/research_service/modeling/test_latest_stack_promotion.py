"""Tests for the latest-stack → model-registry promotion adapter (ADR-004 #14)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from quant_platform.services.research_service.modeling.registry.latest_stack_promotion import (
    ModelRegistration,
    NotPromotableError,
    build_registration,
    promote_to_registry,
)


def _g_evidence(**overrides: object) -> dict[str, object]:
    """A trimmed, realistic Arm-G evidence dict (the fields the adapter reads)."""
    evidence: dict[str, object] = {
        "arm": "long_only_top30_pv_formulaic_streakdial",
        "arm_cli_alias": "G",
        "arm_category": "portfolio_candidate",
        "production_candidate": True,
        "model_version": "ic-weighted-non-negative",
        "feature_set_version": "latest-stack-v1--long_only_top30_pv_formulaic_streakdial",
        "evidence_schema_version": "backtest-latest-stack-realized-v2.1",
        "run_id": "4730187d-36bf-4268-a233-cb3938ee7c5c",
        "git_commit": "9d34da6",
        "saved_at_utc": "2026-05-28T19:17:52.002309+00:00",
        "eligibility": {
            "passed": True,
            "checks": [
                {"name": "fold_negative_ic_streak", "passed": True, "actual": 4.0, "threshold": 6},
            ],
        },
        "eligibility_thresholds": {"name": "portfolio_candidate_v2"},
        "metrics": {
            "slippage_adjusted_sharpe": 1.0886,
            "max_drawdown": -0.0421,
            "fold_negative_ic_streak": 4.0,
            "max_drawdown_during_worst_streak": -0.0029,
            "oos_rolling_ic": 0.2561,
            "ic_60d": 0.0912,
            "total_return": 0.1409,
            "turnover_avg": 0.0048,  # NOT a headline metric — must be dropped
        },
        "universe_fingerprint": {"path": "infra/config/universe_300.json", "sha256": "1f5f"},
        "bars_snapshot_fingerprint": {"files": 1975, "fingerprint": "c2fa"},
    }
    evidence.update(overrides)
    return evidence


def test_build_registration_extracts_identity_and_as_of() -> None:
    reg = build_registration(_g_evidence())
    assert isinstance(reg, ModelRegistration)
    assert reg.strategy_name == "long_only_top30_pv_formulaic_streakdial"
    assert reg.model_version == "ic-weighted-non-negative"
    assert reg.feature_set_version.startswith("latest-stack-v1--")
    assert reg.as_of == datetime(2026, 5, 28, 19, 17, 52, 2309, tzinfo=UTC)


def test_build_registration_packs_audit_metadata() -> None:
    meta = build_registration(_g_evidence()).metadata
    assert meta["source"] == "backtest_latest_stack"
    assert meta["run_id"] == "4730187d-36bf-4268-a233-cb3938ee7c5c"
    assert meta["arm_cli_alias"] == "G"
    # The full gate result rides along as the promotion justification.
    assert meta["eligibility"]["passed"] is True
    assert meta["eligibility_thresholds"]["name"] == "portfolio_candidate_v2"
    # Only headline metrics travel; bulky/irrelevant ones are dropped.
    headline = meta["headline_metrics"]
    assert headline["slippage_adjusted_sharpe"] == pytest.approx(1.0886)
    assert headline["max_drawdown_during_worst_streak"] == pytest.approx(-0.0029)
    assert "turnover_avg" not in headline


def test_as_of_override_takes_precedence() -> None:
    override = datetime(2030, 1, 1, tzinfo=UTC)
    assert build_registration(_g_evidence(), as_of=override).as_of == override


def test_rejects_failed_eligibility() -> None:
    evidence = _g_evidence(eligibility={"passed": False, "checks": []})
    with pytest.raises(NotPromotableError, match="eligibility gate"):
        build_registration(evidence)


def test_rejects_baseline_category() -> None:
    evidence = _g_evidence(arm_category="research_ranker_baseline", production_candidate=False)
    with pytest.raises(NotPromotableError, match="portfolio_candidate"):
        build_registration(evidence)


def test_rejects_non_production_candidate() -> None:
    evidence = _g_evidence(production_candidate=False)
    with pytest.raises(NotPromotableError, match="production_candidate"):
        build_registration(evidence)


def test_rejects_missing_identity_field() -> None:
    evidence = _g_evidence()
    del evidence["model_version"]
    with pytest.raises(NotPromotableError, match="model_version"):
        build_registration(evidence)


def test_missing_saved_at_requires_explicit_as_of() -> None:
    evidence = _g_evidence()
    del evidence["saved_at_utc"]
    with pytest.raises(NotPromotableError, match="saved_at_utc"):
        build_registration(evidence)
    # ...but passing as_of explicitly works.
    override = datetime(2026, 6, 1, tzinfo=UTC)
    assert build_registration(evidence, as_of=override).as_of == override


class _FakeAsyncRegistry:
    """Minimal async registry capturing register_model calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def register_model(
        self,
        *,
        strategy_name: str,
        model_version: str,
        feature_set_version: str,
        as_of: datetime,
        metadata: dict[str, object] | None = None,
    ) -> object:
        record = SimpleNamespace(
            model_id=uuid.uuid4(),
            strategy_name=strategy_name,
            model_version=model_version,
            feature_set_version=feature_set_version,
            created_at=as_of,
            metadata=dict(metadata or {}),
            active=True,
        )
        self.calls.append(vars(record))
        return record


def test_promote_to_registry_registers_eligible_arm() -> None:
    registry = _FakeAsyncRegistry()
    record = asyncio.run(promote_to_registry(registry, _g_evidence()))
    assert len(registry.calls) == 1
    assert record.strategy_name == "long_only_top30_pv_formulaic_streakdial"
    assert record.active is True
    assert registry.calls[0]["metadata"]["source"] == "backtest_latest_stack"


def test_promote_to_registry_never_touches_registry_for_ineligible_arm() -> None:
    registry = _FakeAsyncRegistry()
    evidence = _g_evidence(eligibility={"passed": False, "checks": []})
    with pytest.raises(NotPromotableError):
        asyncio.run(promote_to_registry(registry, evidence))
    assert registry.calls == []  # rejected before persistence
