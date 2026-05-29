from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from quant_platform.config import LLMSettings, PlatformSettings, StorageSettings
from quant_platform.core.domain.production import RuntimeHeartbeat, SignalGateStatus
from quant_platform.services.governance_service import alpha
from quant_platform.services.research_service.modeling.registry.model_registry import (
    RegisteredModel,
)

_NOW = datetime(2026, 1, 2, tzinfo=UTC)


def _passing_gate(signal_name: str = "xsec", signal_type: str = "xgboost") -> SignalGateStatus:
    return SignalGateStatus(
        signal_name=signal_name,
        signal_type=signal_type,
        as_of=_NOW,
        observations=30,
        rolling_ic=0.08,
        negative_streak=0,
        max_drawdown=-0.02,
        max_turnover=0.2,
        min_observations=20,
        min_ic=0.05,
        max_negative_streak=3,
        drawdown_limit=-0.10,
        turnover_limit=1.0,
    )


def _model(version: str = "v1") -> RegisteredModel:
    return RegisteredModel(
        model_id=uuid.uuid4(),
        strategy_name="xsec",
        model_version=version,
        feature_set_version="features-v1",
        created_at=_NOW,
    )


class _Registry:
    def __init__(self) -> None:
        self.active = _model("active-v1")
        self.registered: list[dict[str, Any]] = []
        self.rollback_target: str | None = None

    async def get_active_model(self, _strategy_name: str) -> RegisteredModel:
        return self.active

    async def register_model(self, **kwargs: Any) -> RegisteredModel:
        self.registered.append(kwargs)
        return _model(str(kwargs["model_version"]))

    async def rollback_to_version(
        self, _strategy_name: str, target_version: str
    ) -> RegisteredModel:
        self.rollback_target = target_version
        return _model(target_version)


class _PerformanceRepo:
    def __init__(self) -> None:
        self.heartbeats: list[RuntimeHeartbeat] = []

    async def save_runtime_heartbeat(self, heartbeat: RuntimeHeartbeat) -> None:
        self.heartbeats.append(heartbeat)


@pytest.mark.asyncio
async def test_alpha_assert_accepts_registered_xgboost_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()

    async def _gate(_settings: PlatformSettings, **kwargs: Any) -> SignalGateStatus:
        return _passing_gate(kwargs["signal_name"], kwargs["signal_type"])

    monkeypatch.setattr(alpha, "signal_gate_status", _gate)
    monkeypatch.setattr(alpha, "build_model_registry", lambda _dsn: registry)

    booster = tmp_path / "model.json"
    booster.write_text("booster", encoding="utf-8")
    digest = hashlib.sha256(booster.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "booster_path": "model.json",
                "booster_sha256": digest,
                "metrics": {
                    "validation_ic": 0.06,
                    "train_groups": 252,
                    "validation_groups": 20,
                    "feature_coverage": 0.97,
                },
            }
        ),
        encoding="utf-8",
    )
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="postgresql+psycopg://u:p@localhost/db"),
    )

    result = await alpha.alpha_assert(
        settings,
        signal_name="xsec",
        signal_type="xgboost",
        as_of=_NOW,
        artifact_manifest=manifest,
    )

    assert result["passed"] is True
    checks = {check["name"]: check["passed"] for check in result["checks"]}
    assert checks["signal_gate_passed"]
    assert checks["active_model_registered"]
    assert checks["artifact_hash_verified"]


@pytest.mark.asyncio
async def test_alpha_assert_validates_text_manifest_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    registry.active = RegisteredModel(
        model_id=uuid.uuid4(),
        strategy_name="text",
        model_version="text-v1",
        feature_set_version="paper-alpha-catalyst-v10",
        created_at=_NOW,
    )

    async def _gate(_settings: PlatformSettings, **kwargs: Any) -> SignalGateStatus:
        return _passing_gate(kwargs["signal_name"], kwargs["signal_type"])

    monkeypatch.setattr(alpha, "signal_gate_status", _gate)
    monkeypatch.setattr(alpha, "build_model_registry", lambda _dsn: registry)
    weights = {
        "text_sentiment_21d_decay": 0.5,
        "guidance_direction_21d_decay": 0.3,
        "revenue_revision_direction_21d_decay": 0.2,
    }
    feature_names = tuple(weights)
    schema_hash = alpha.ordered_feature_schema_hash(feature_names)
    manifest = tmp_path / "text_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "signal_type": "text",
                "model_version": "text-v1",
                "feature_set_version": "paper-alpha-catalyst-v10",
                "feature_names": list(feature_names),
                "feature_schema_hash": schema_hash,
                "weights": weights,
                "llm_model": "claude-sonnet-4-6",
                "prompt_version": "v1",
            }
        ),
        encoding="utf-8",
    )
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="postgresql+psycopg://u:p@localhost/db"),
        llm=LLMSettings(text_feature_weights=weights),
    )

    result = await alpha.alpha_assert(
        settings,
        signal_name="text",
        signal_type="text",
        as_of=_NOW,
        artifact_manifest=manifest,
    )

    checks = {check["name"]: check["passed"] for check in result["checks"]}
    assert result["passed"] is True
    assert checks["text_weights_match_manifest"]
    assert checks["text_manifest_feature_schema_hash"]


@pytest.mark.asyncio
async def test_alpha_promote_and_rollback_write_audit_heartbeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    repo = _PerformanceRepo()
    monkeypatch.setattr(alpha, "build_model_registry", lambda _dsn: registry)
    monkeypatch.setattr(alpha, "build_performance_repository", lambda _dsn: repo)
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="postgresql+psycopg://u:p@localhost/db"),
    )

    promoted = await alpha.alpha_promote(
        settings,
        signal_name="xsec",
        signal_type="xgboost",
        model_version="v2",
        feature_set_version="features-v2",
        engine_version="engine-v1",
        artifact_manifest=Path("manifest.json"),
        rollback_target="v1",
        as_of=_NOW,
    )
    rolled_back = await alpha.alpha_rollback(
        settings,
        signal_name="xsec",
        target_version="v1",
        as_of=_NOW,
    )

    assert promoted["promoted"] is True
    assert promoted["model_version"] == "v2"
    assert registry.registered[0]["metadata"]["engine_version"] == "engine-v1"
    assert rolled_back["rolled_back"] is True
    assert registry.rollback_target == "v1"
    assert [heartbeat.component for heartbeat in repo.heartbeats] == [
        "alpha:xgboost:xsec",
        "alpha:rollback:xsec",
    ]


@pytest.mark.asyncio
async def test_alpha_promote_attaches_evidence_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    repo = _PerformanceRepo()
    monkeypatch.setattr(alpha, "build_model_registry", lambda _dsn: registry)
    monkeypatch.setattr(alpha, "build_performance_repository", lambda _dsn: repo)
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="postgresql+psycopg://u:p@localhost/db"),
    )

    await alpha.alpha_promote(
        settings,
        signal_name="long_only_top30_pv_formulaic_streakdial",
        signal_type="xgboost",
        model_version="ic-weighted-non-negative",
        feature_set_version="latest-stack-v1--g",
        engine_version="engine-v1",
        artifact_manifest=None,
        rollback_target="",
        as_of=_NOW,
        evidence_metadata={"source": "backtest_latest_stack", "eligibility": {"passed": True}},
    )

    metadata = registry.registered[0]["metadata"]
    # The adapter's provenance rides under a single "evidence" key...
    assert metadata["evidence"]["source"] == "backtest_latest_stack"
    assert metadata["evidence"]["eligibility"]["passed"] is True
    # ...without disturbing the existing alpha block / engine_version.
    assert metadata["engine_version"] == "engine-v1"
    assert metadata["alpha"]["signal_type"] == "xgboost"


@pytest.mark.asyncio
async def test_alpha_promote_omits_evidence_key_when_not_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    repo = _PerformanceRepo()
    monkeypatch.setattr(alpha, "build_model_registry", lambda _dsn: registry)
    monkeypatch.setattr(alpha, "build_performance_repository", lambda _dsn: repo)
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="postgresql+psycopg://u:p@localhost/db"),
    )

    await alpha.alpha_promote(
        settings,
        signal_name="xsec",
        signal_type="xgboost",
        model_version="v2",
        feature_set_version="features-v2",
        engine_version="engine-v1",
        artifact_manifest=None,
        rollback_target="v1",
        as_of=_NOW,
    )

    assert "evidence" not in registry.registered[0]["metadata"]


@pytest.mark.asyncio
async def test_alpha_promote_requires_postgres_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QP__STORAGE__POSTGRES_DSN", raising=False)
    with pytest.raises(RuntimeError, match="alpha promote requires"):
        await alpha.alpha_promote(
            PlatformSettings(_env_file=None),
            signal_name="xsec",
            signal_type="xgboost",
            model_version="v2",
            feature_set_version="features-v2",
            engine_version="engine-v1",
            artifact_manifest=None,
            rollback_target="v1",
            as_of=_NOW,
        )


def test_alpha_ramp_levels() -> None:
    settings = PlatformSettings(_env_file=None)

    assert alpha.alpha_ramp(settings, clean_live_days=0)["ramp_level"] == "0.01"
    assert alpha.alpha_ramp(settings, clean_live_days=20)["ramp_level"] == "0.10"
    assert alpha.alpha_ramp(settings, clean_live_days=60)["ramp_level"] == "0.20"
