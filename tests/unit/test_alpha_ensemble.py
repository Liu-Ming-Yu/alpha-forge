from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.bootstrap.signal_models import build_default_primary_signal_model
from quant_platform.config import (
    AlphaSettings,
    BoostingSettings,
    LLMSettings,
    PlatformSettings,
    StorageSettings,
)
from quant_platform.core.domain.research import (
    FeatureProductionState,
    FeatureVector,
    RunStatus,
    RunType,
    StrategyRun,
)
from quant_platform.infrastructure.performance import InMemoryPerformanceRepository
from quant_platform.infrastructure.repositories.signal_contributions import (
    InMemorySignalContributionRepository,
)
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.governance_service.llm_live_startup import (
    write_llm_live_startup_assertion,
)
from quant_platform.services.research_service.text.model_manifest import (
    write_text_model_manifest,
)
from quant_platform.services.signal_service.controllers import GenerateSignalsControllerImpl
from quant_platform.services.signal_service.ensemble import (
    EnsembleSignalModel,
    MissingPromotedSignalSourceError,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.session import create_paper_session

if TYPE_CHECKING:
    from pathlib import Path


class _Bus:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, event):  # type: ignore[no-untyped-def]
        self.events.append(event)


def _run() -> StrategyRun:
    now = datetime(2026, 4, 29, tzinfo=UTC)
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="alpha_test",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=now,
        started_at=now,
    )


def test_ensemble_blends_and_clamps_with_contributions() -> None:
    run = _run()
    vector = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        as_of=datetime(2026, 4, 29, tzinfo=UTC),
        feature_set_version="1.0.0",
        features={"momentum_1m": 2.0, "text_sentiment": 1.0},
        strategy_run_id=run.run_id,
    )
    model = EnsembleSignalModel(
        sources={
            "classical": LinearWeightSignalModel({"momentum_1m": 1.0}, "classical-v1"),
            "text": LinearWeightSignalModel({"text_sentiment": 1.0}, "text-v1"),
        },
        source_weights={"classical": 0.8, "text": 0.2},
        mode="paper",
        max_non_classical_weight=0.2,
        text_required_features={"text_sentiment"},
    )

    scores = model.score([vector], run)

    assert scores[0].score == pytest.approx(1.0)
    assert {row.source for row in model.last_contributions} == {"classical", "text"}
    assert sum(row.blend_weight for row in model.last_contributions) == pytest.approx(1.0)


def test_ensemble_live_fails_when_promoted_source_missing_model() -> None:
    run = _run()
    vector = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        as_of=datetime(2026, 4, 29, tzinfo=UTC),
        feature_set_version="1.0.0",
        features={"momentum_1m": 0.4},
        strategy_run_id=run.run_id,
    )
    model = EnsembleSignalModel(
        sources={"classical": LinearWeightSignalModel({"momentum_1m": 1.0})},
        source_weights={"classical": 0.8, "xgboost": 0.2},
        mode="live",
        max_non_classical_weight=0.05,
        fail_closed=True,
    )

    with pytest.raises(MissingPromotedSignalSourceError):
        model.score([vector], run)


def test_paper_ensemble_requires_promoted_xgboost_manifest() -> None:
    settings = PlatformSettings(
        _env_file=None,
        alpha=AlphaSettings(
            ensemble_mode="paper",
            source_weights={"classical": 0.7, "xgboost": 0.3},
        ),
    )

    with pytest.raises(RuntimeError, match="ARTIFACT_MANIFEST"):
        create_paper_session(settings)


def test_paper_ensemble_requires_boosting_feature_versions(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_type": "xgboost_ranker",
                "model_version": "xgb-paper",
                "feature_set_version": "features-v1",
                "booster_path": "model.json",
                "feature_names": ["quality_alpha"],
                "feature_schema_hash": ordered_feature_schema_hash(("quality_alpha",)),
                "xgboost_version": "test",
                "objective": "rank:pairwise",
                "device": "cpu",
                "trained_at": "2026-01-01T00:00:00+00:00",
                "metrics_path": "metrics.json",
            }
        ),
        encoding="utf-8",
    )
    settings = PlatformSettings(
        _env_file=None,
        alpha=AlphaSettings(
            ensemble_mode="paper",
            source_weights={"classical": 0.7, "xgboost": 0.3},
        ),
        boosting=BoostingSettings(artifact_manifest=str(manifest), device="cpu"),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )

    with pytest.raises(RuntimeError, match="feature_versions"):
        create_paper_session(settings)


def test_paper_ensemble_requires_feature_audit_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_type": "xgboost_ranker",
                "model_version": "xgb-paper",
                "feature_set_version": "features-v1",
                "booster_path": "model.json",
                "feature_names": ["quality_alpha"],
                "feature_versions": {"quality_alpha": "v1"},
                "feature_schema_hash": ordered_feature_schema_hash(("quality_alpha",)),
                "xgboost_version": "test",
                "objective": "rank:pairwise",
                "device": "cpu",
                "trained_at": "2026-01-01T00:00:00+00:00",
                "metrics_path": "metrics.json",
            }
        ),
        encoding="utf-8",
    )
    settings = PlatformSettings(
        _env_file=None,
        alpha=AlphaSettings(
            ensemble_mode="paper",
            source_weights={"classical": 0.7, "xgboost": 0.3},
        ),
        boosting=BoostingSettings(artifact_manifest=str(manifest), device="cpu"),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )

    with pytest.raises(RuntimeError, match="no feature audit manifests"):
        create_paper_session(settings)


def test_paper_text_ensemble_uses_audited_text_with_llm_live_off(tmp_path: Path) -> None:
    feature_weights = {
        ("v8_neg_text_sentiment_vol_compression_fresh5_plus_catalyst_specificity_21d"): 0.40,
        "v8_neg_text_sentiment_vol_compression_fresh5_plus_event_surprise_21d": 0.35,
        ("v8_neg_text_sentiment_vol_compression_fresh5_plus_catalyst_risk_specificity_21d"): 0.25,
    }
    for feature_name in feature_weights:
        _write_feature_audit_manifest(
            tmp_path,
            feature_name=feature_name,
            feature_version="paper-alpha-catalyst-v10",
        )
    settings = PlatformSettings(
        _env_file=None,
        alpha=AlphaSettings(
            ensemble_mode="paper",
            source_weights={"classical": 0.95, "text": 0.05},
        ),
        llm=LLMSettings(
            shadow_mode_enabled=True,
            live_mode_enabled=False,
            text_feature_weights=feature_weights,
            text_feature_set_version="paper-alpha-catalyst-v10",
        ),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )

    session = create_paper_session(settings)

    assert session.signal_ctrl is not None
    assert session.signal_ctrl._model.model_version == "ensemble-paper-v1"  # noqa: SLF001
    assert settings.llm.live_mode_enabled is False


def test_paper_ensemble_accepts_composite_audits_with_source_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeXGBoostModel:
        model_version = "xgb-paper"

        def __init__(
            self,
            manifest_path: object,
            *,
            device: object = None,
            require_gpu: bool = False,
        ) -> None:
            del manifest_path, device, require_gpu
            pass

    text_feature = "text_alpha"
    event_feature = "event_alpha"
    intraday_feature = "intraday_alpha"
    composite_feature_set = "paper-alpha-composite-v1"
    text_feature_set = "text-v10"
    event_feature_set = "event-v2"
    intraday_feature_set = "intraday-v2"
    for feature_name, feature_version in (
        (text_feature, text_feature_set),
        (event_feature, event_feature_set),
        (intraday_feature, intraday_feature_set),
    ):
        _write_feature_audit_manifest(
            tmp_path,
            feature_name=feature_name,
            feature_version=feature_version,
            feature_set_version=composite_feature_set,
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_type": "xgboost_ranker",
                "model_version": "xgb-paper",
                "feature_set_version": composite_feature_set,
                "booster_path": "model.json",
                "feature_names": [text_feature],
                "feature_versions": {text_feature: composite_feature_set},
                "feature_schema_hash": ordered_feature_schema_hash((text_feature,)),
                "xgboost_version": "test",
                "objective": "rank:pairwise",
                "device": "cpu",
                "trained_at": "2026-01-01T00:00:00+00:00",
                "metrics_path": "metrics.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "quant_platform.services.research_service.boosting.XGBoostRankSignalModel",
        _FakeXGBoostModel,
    )
    settings = PlatformSettings(
        _env_file=None,
        alpha=AlphaSettings(
            ensemble_mode="paper",
            source_weights={
                "classical": 0.70,
                "xgboost": 0.15,
                "text": 0.05,
                "event": 0.05,
                "intraday": 0.05,
            },
            promoted_feature_set_version=composite_feature_set,
            event_feature_weights={event_feature: 1.0},
            event_feature_set_version=event_feature_set,
            intraday_feature_weights={intraday_feature: 1.0},
            intraday_feature_set_version=intraday_feature_set,
        ),
        boosting=BoostingSettings(artifact_manifest=str(manifest), device="cpu"),
        llm=LLMSettings(
            shadow_mode_enabled=True,
            live_mode_enabled=False,
            text_feature_weights={text_feature: 1.0},
            text_feature_set_version=text_feature_set,
        ),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )

    session = create_paper_session(settings)

    assert session.signal_ctrl is not None
    assert session.signal_ctrl._model.model_version == "ensemble-paper-v1"  # noqa: SLF001


def test_llm_live_rehearsal_session_uses_paper_audited_text(tmp_path: Path) -> None:
    as_of = datetime(2026, 4, 29, tzinfo=UTC)
    feature_name = "live_text_alpha"
    feature_set = "text-live-v1"
    _write_feature_audit_manifest(
        tmp_path,
        feature_name=feature_name,
        feature_version=feature_set,
    )
    card_dir = tmp_path / "cards"
    card_dir.mkdir()
    (card_dir / f"{feature_name}.json").write_text(
        json.dumps({"feature": feature_name, "state": "paper"}, sort_keys=True),
        encoding="utf-8",
    )
    campaign = tmp_path / "campaign_manifest.json"
    campaign.write_text("{}", encoding="utf-8")
    extraction = tmp_path / "text_extraction_manifest.json"
    extraction.write_text("{}", encoding="utf-8")

    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
        alpha=AlphaSettings(
            ensemble_mode="live",
            source_weights={"classical": 0.99, "text": 0.01},
            max_non_classical_weight=0.01,
        ),
        llm=LLMSettings(
            shadow_mode_enabled=True,
            live_mode_enabled=True,
            live_rehearsal_enabled=True,
            text_feature_weights={feature_name: 1.0},
            text_feature_versions={feature_name: feature_set},
            text_feature_set_version=feature_set,
            text_feature_card_dir=str(card_dir),
        ),
    )
    manifest = write_text_model_manifest(
        output_root=tmp_path,
        model_version="text-v1",
        feature_set_version=feature_set,
        feature_names=(feature_name,),
        weights={feature_name: 1.0},
        provider=settings.llm.provider,
        llm_model=settings.llm.model,
        prompt_version=settings.llm.text_prompt_version,
        campaign_manifest=campaign,
        source_data_manifest=None,
        extraction_manifest=extraction,
        feature_card_dir=card_dir,
        created_at=as_of,
    )
    settings = settings.model_copy(
        update={"llm": settings.llm.model_copy(update={"text_model_manifest": str(manifest)})}
    )
    write_llm_live_startup_assertion(
        settings,
        candidate_payload={
            "profile": "llm_live_rehearsal",
            "passed": True,
            "next_allowed_mode": "llm_live_rehearsal",
        },
        as_of=as_of,
    )

    session = create_paper_session(settings, clock=FakeClock(as_of))

    assert session.signal_ctrl is not None
    assert session.signal_ctrl._model.model_version == "ensemble-live-v1"  # noqa: SLF001


def test_paper_ensemble_caps_non_classical_at_aggressive_paper_limit() -> None:
    run = _run()
    vector = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        as_of=datetime(2026, 4, 29, tzinfo=UTC),
        feature_set_version="1.0.0",
        features={"momentum_1m": 1.0, "text_sentiment": 1.0},
        strategy_run_id=run.run_id,
    )
    model = EnsembleSignalModel(
        sources={
            "classical": LinearWeightSignalModel({"momentum_1m": 1.0}, "classical-v1"),
            "text": LinearWeightSignalModel({"text_sentiment": 1.0}, "text-v1"),
        },
        source_weights={"classical": 0.5, "text": 0.5},
        mode="paper",
        max_non_classical_weight=0.30,
        text_required_features={"text_sentiment"},
    )

    model.score([vector], run)

    weights = {row.source: row.blend_weight for row in model.last_contributions}
    assert weights["text"] == pytest.approx(0.30)
    assert weights["classical"] == pytest.approx(0.70)


def test_paper_ensemble_caps_all_non_classical_sources() -> None:
    run = _run()
    vector = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        as_of=datetime(2026, 4, 29, tzinfo=UTC),
        feature_set_version="paper-alpha-composite-v1",
        features={
            "momentum_1m": 1.0,
            "text_alpha": 1.0,
            "event_alpha": 1.0,
            "intraday_alpha": 1.0,
        },
        strategy_run_id=run.run_id,
    )
    model = EnsembleSignalModel(
        sources={
            "classical": LinearWeightSignalModel({"momentum_1m": 1.0}, "classical-v1"),
            "text": LinearWeightSignalModel({"text_alpha": 1.0}, "text-v1"),
            "event": LinearWeightSignalModel({"event_alpha": 1.0}, "event-v1"),
            "intraday": LinearWeightSignalModel({"intraday_alpha": 1.0}, "intraday-v1"),
        },
        source_weights={"classical": 0.4, "text": 0.2, "event": 0.2, "intraday": 0.2},
        mode="paper",
        max_non_classical_weight=0.30,
        required_features_by_source={
            "text": {"text_alpha"},
            "event": {"event_alpha"},
            "intraday": {"intraday_alpha"},
        },
    )

    model.score([vector], run)

    weights = {row.source: row.blend_weight for row in model.last_contributions}
    assert weights["classical"] == pytest.approx(0.70)
    assert weights["text"] == pytest.approx(0.10)
    assert weights["event"] == pytest.approx(0.10)
    assert weights["intraday"] == pytest.approx(0.10)


def test_paper_ensemble_fails_closed_when_event_feature_missing() -> None:
    run = _run()
    vector = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        as_of=datetime(2026, 4, 29, tzinfo=UTC),
        feature_set_version="paper-alpha-composite-v1",
        features={"momentum_1m": 1.0},
        strategy_run_id=run.run_id,
    )
    model = EnsembleSignalModel(
        sources={
            "classical": LinearWeightSignalModel({"momentum_1m": 1.0}, "classical-v1"),
            "event": LinearWeightSignalModel({"event_alpha": 1.0}, "event-v1"),
        },
        source_weights={"classical": 0.95, "event": 0.05},
        mode="paper",
        max_non_classical_weight=0.30,
        required_features_by_source={"event": {"event_alpha"}},
    )

    with pytest.raises(MissingPromotedSignalSourceError, match="event source missing"):
        model.score([vector], run)


def test_paper_target_classical_text_weights_only() -> None:
    run = _run()
    vector = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        as_of=datetime(2026, 4, 29, tzinfo=UTC),
        feature_set_version="paper-alpha-composite-v1",
        features={
            "momentum_1m": 1.0,
            "text_alpha": 1.0,
            "xgb_alpha": 1.0,
            "event_alpha": 1.0,
            "intraday_alpha": 1.0,
        },
        strategy_run_id=run.run_id,
    )
    model = EnsembleSignalModel(
        sources={
            "classical": LinearWeightSignalModel({"momentum_1m": 1.0}, "classical-v1"),
            "text": LinearWeightSignalModel({"text_alpha": 1.0}, "text-v1"),
            "xgboost": LinearWeightSignalModel({"xgb_alpha": 1.0}, "xgb-v1"),
            "event": LinearWeightSignalModel({"event_alpha": 1.0}, "event-v1"),
            "intraday": LinearWeightSignalModel({"intraday_alpha": 1.0}, "intraday-v1"),
        },
        source_weights={
            "classical": 0.95,
            "text": 0.05,
            "xgboost": 0.0,
            "event": 0.0,
            "intraday": 0.0,
        },
        mode="paper",
        max_non_classical_weight=0.05,
        required_features_by_source={"text": {"text_alpha"}},
    )

    model.score([vector], run)

    weights = {row.source: row.blend_weight for row in model.last_contributions}
    assert set(weights) == {"classical", "text"}
    assert weights["classical"] == pytest.approx(0.95)
    assert weights["text"] == pytest.approx(0.05)


def test_live_llm_rehearsal_fails_when_one_text_feature_missing() -> None:
    run = _run()
    model = build_default_primary_signal_model(_llm_rehearsal_settings())
    vector = _text_vector(run, {"text_alpha_a": 0.4, "momentum_1m": 0.2})

    with pytest.raises(MissingPromotedSignalSourceError, match="text_alpha_b"):
        model.score([vector], run)


def test_live_llm_rehearsal_fails_when_text_feature_non_finite() -> None:
    run = _run()
    model = build_default_primary_signal_model(_llm_rehearsal_settings())
    vector = _text_vector(
        run,
        {"text_alpha_a": 0.4, "text_alpha_b": float("nan"), "momentum_1m": 0.2},
    )

    with pytest.raises(MissingPromotedSignalSourceError, match="non-finite"):
        model.score([vector], run)


def test_live_llm_rehearsal_all_text_features_present_passes() -> None:
    run = _run()
    model = build_default_primary_signal_model(_llm_rehearsal_settings())
    vector = _text_vector(
        run,
        {"text_alpha_a": 0.4, "text_alpha_b": 0.3, "momentum_1m": 0.2},
    )

    scores = model.score([vector], run)

    assert len(scores) == 1
    assert scores[0].confidence > 0.0


@pytest.mark.asyncio
async def test_generate_signals_persists_ensemble_contributions() -> None:
    repo = InMemorySignalContributionRepository()
    bus = _Bus()
    run = _run()
    model = EnsembleSignalModel(
        sources={"classical": LinearWeightSignalModel({"momentum_1m": 0.5})},
        source_weights={"classical": 1.0},
        mode="paper",
        max_non_classical_weight=0.05,
    )
    ctrl = GenerateSignalsControllerImpl(model, bus, signal_contribution_repo=repo)
    instrument_id = uuid.uuid4()

    scores = await ctrl.generate(
        {instrument_id: {"momentum_1m": 1.0}},
        run,
        datetime(2026, 4, 29, tzinfo=UTC),
    )
    rows = await repo.list_signal_contributions(score_id=scores[0].score_id)

    assert len(rows) == 1
    assert rows[0].source == "classical"


@pytest.mark.asyncio
async def test_generate_signals_persists_text_prediction_evidence() -> None:
    repo = InMemoryPerformanceRepository()
    bus = _Bus()
    run = _run()
    model = EnsembleSignalModel(
        sources={
            "classical": LinearWeightSignalModel({"momentum_1m": 0.5}, "classical-v1"),
            "text": LinearWeightSignalModel({"text_alpha": 1.0}, "text-v1"),
        },
        source_weights={
            "classical": 0.95,
            "text": 0.05,
            "xgboost": 0.0,
            "event": 0.0,
            "intraday": 0.0,
        },
        mode="paper",
        max_non_classical_weight=0.05,
        text_required_features={"text_alpha"},
    )
    ctrl = GenerateSignalsControllerImpl(model, bus, prediction_evidence_repo=repo)
    instrument_id = uuid.uuid4()
    as_of = datetime(2026, 4, 29, tzinfo=UTC)

    await ctrl.generate(
        {instrument_id: {"momentum_1m": 1.0, "text_alpha": 0.8}},
        run,
        as_of,
    )

    rows = await repo.list_prediction_results(source="text")
    classical_rows = await repo.list_prediction_results(source="classical")
    evidence = await repo.forecast_evidence("text", as_of=as_of)

    assert len(rows) == 1
    assert classical_rows == []
    assert rows[0].calibration_bucket == "rank_score_uncalibrated"
    assert rows[0].feature_schema_hash == ordered_feature_schema_hash(("text_alpha",))
    assert rows[0].metadata["expected_return_semantics"] == "rank_score_proxy"
    assert rows[0].metadata["feature_names"] == ["text_alpha"]
    assert evidence.passed is True


def _llm_rehearsal_settings() -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        alpha=AlphaSettings(
            ensemble_mode="live",
            source_weights={
                "classical": 0.99,
                "text": 0.01,
                "xgboost": 0.0,
                "event": 0.0,
                "intraday": 0.0,
            },
            max_non_classical_weight=0.01,
            require_promotion_gate=False,
        ),
        llm=LLMSettings(
            shadow_mode_enabled=True,
            live_mode_enabled=True,
            live_rehearsal_enabled=True,
            text_feature_weights={"text_alpha_a": 0.6, "text_alpha_b": 0.4},
            text_feature_set_version="text-live-v1",
        ),
    )


def _text_vector(run: StrategyRun, features: dict[str, float]) -> FeatureVector:
    return FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        as_of=datetime(2026, 4, 29, tzinfo=UTC),
        feature_set_version="text-live-v1",
        features=features,
        strategy_run_id=run.run_id,
    )


def _write_feature_audit_manifest(
    root: Path,
    *,
    feature_name: str,
    feature_version: str,
    feature_set_version: str | None = None,
) -> None:
    audit_id = uuid.uuid4()
    generated_at = datetime(2026, 4, 29, tzinfo=UTC)
    audit_dir = (
        root / "research" / "feature_audits" / feature_name / feature_version / str(audit_id)
    )
    audit_dir.mkdir(parents=True)
    payload = {
        "audit_id": str(audit_id),
        "feature": {
            "name": feature_name,
            "version": feature_version,
            "owner": "research",
            "economic_thesis": "test",
            "source_datasets": ["unit"],
            "required_lags": ["available_at <= as_of"],
            "valid_universe": "unit",
            "expected_sign": "positive",
            "horizon_days": 21,
            "expected_turnover": "low",
            "state": FeatureProductionState.PAPER.value,
            "failure_modes": ["unit"],
            "risk_exposures": ["unit"],
        },
        "feature_set_version": feature_set_version or feature_version,
        "generated_at": generated_at.isoformat(),
        "sample_start": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "sample_end": generated_at.isoformat(),
        "sample_count": 30,
        "passed": True,
        "gate_results": {},
        "metrics": {},
        "blockers": [],
        "artifacts": {},
        "schema_hash": "unit",
        "code_commit": "unit",
    }
    (audit_dir / "feature_audit_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
