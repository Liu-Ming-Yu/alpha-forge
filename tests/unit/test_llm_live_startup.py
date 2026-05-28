from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.config import (
    AlphaSettings,
    BrokerSettings,
    ExecutionSettings,
    LLMSettings,
    PlatformSettings,
    StorageSettings,
)
from quant_platform.core.domain.production import ForecastEvidence, ProductionProfile
from quant_platform.services.governance_service.llm_live_startup import (
    assert_llm_live_startup_allowed,
    build_llm_live_evidence_checks,
    write_llm_live_startup_assertion,
)
from quant_platform.services.research_service.text.model_manifest import (
    write_text_model_manifest,
)

_AS_OF = datetime(2026, 5, 14, 15, 0, tzinfo=UTC)
_FEATURE = "live_text_alpha"
_FEATURE_SET = "text-live-v1"


def _live_settings(
    tmp_path: Path,
    *,
    as_of: datetime = _AS_OF,
    cap: Decimal = Decimal("0.01"),
) -> tuple[PlatformSettings, Path, Path]:
    object_root = tmp_path / "objects"
    object_root.mkdir(parents=True, exist_ok=True)
    card_dir = tmp_path / "cards"
    card_dir.mkdir()
    card_path = card_dir / f"{_FEATURE}.json"
    card_path.write_text(
        json.dumps(
            {
                "feature": _FEATURE,
                "owner": "research",
                "state": "live",
                "schema_hash": ordered_feature_schema_hash((_FEATURE,)),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(object_root)),
        alpha=AlphaSettings(
            ensemble_mode="live",
            source_weights={"classical": 0.99, "text": 0.01},
            max_non_classical_weight=float(cap),
            live_ramp_initial=cap,
        ),
        llm=LLMSettings(
            live_mode_enabled=True,
            shadow_mode_enabled=True,
            text_feature_weights={_FEATURE: 1.0},
            text_feature_versions={_FEATURE: _FEATURE_SET},
            text_feature_set_version=_FEATURE_SET,
            text_feature_card_dir=str(card_dir),
        ),
    )
    campaign = tmp_path / "campaign_manifest.json"
    campaign.write_text("{}", encoding="utf-8")
    extraction = tmp_path / "text_extraction_manifest.json"
    extraction.write_text("{}", encoding="utf-8")
    manifest = write_text_model_manifest(
        output_root=object_root,
        model_version="text-v1",
        feature_set_version=_FEATURE_SET,
        feature_names=(_FEATURE,),
        weights={_FEATURE: 1.0},
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
    return settings, manifest, card_path


def _write_live_feature_audit(
    settings: PlatformSettings,
    *,
    as_of: datetime = _AS_OF,
    state: str = "live",
) -> None:
    audit_dir = (
        Path(settings.storage.object_store_root)
        / "research"
        / "feature_audits"
        / _FEATURE
        / _FEATURE_SET
        / str(uuid.uuid4())
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit_id": str(uuid.uuid4()),
        "generated_at": as_of.isoformat(),
        "sample_start": (as_of - timedelta(days=30)).isoformat(),
        "sample_end": as_of.isoformat(),
        "feature_set_version": _FEATURE_SET,
        "feature": {"name": _FEATURE, "version": _FEATURE_SET, "state": state},
        "passed": True,
        "metrics": {"rolling_ic": 0.08},
        "gate_results": {"ic_gate": True},
        "schema_hash": ordered_feature_schema_hash((_FEATURE,)),
        "blockers": [],
    }
    (audit_dir / "feature_audit_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _green_text_forecast(settings: PlatformSettings) -> ForecastEvidence:
    return ForecastEvidence(
        source="text",
        model_version="text-v1",
        as_of=_AS_OF,
        horizon="21d",
        observations=25,
        mean_confidence=0.70,
        latest_prediction_at=_AS_OF,
        stale_after=timedelta(hours=24),
        feature_schema_hashes=(
            ordered_feature_schema_hash(tuple(settings.llm.text_feature_weights)),
        ),
    )


def test_live_llm_evidence_requires_explicit_manifest(tmp_path: Path) -> None:
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
        alpha=AlphaSettings(ensemble_mode="live"),
        llm=LLMSettings(live_mode_enabled=True),
    )

    checks = build_llm_live_evidence_checks(settings, as_of=_AS_OF)

    assert checks[-1].name == "llm_live_text_model_manifest_present"
    assert checks[-1].passed is False


def test_live_llm_startup_accepts_fresh_matching_assertion(tmp_path: Path) -> None:
    settings, _manifest, _card = _live_settings(tmp_path)
    write_llm_live_startup_assertion(
        settings,
        candidate_payload={"passed": True, "next_allowed_mode": "live_ramp_initial"},
        as_of=_AS_OF,
    )

    assert_llm_live_startup_allowed(settings, now=_AS_OF + timedelta(minutes=5))


def test_live_llm_rehearsal_accepts_paper_audited_assertion(tmp_path: Path) -> None:
    settings, _manifest, _card = _live_settings(tmp_path)
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(update={"live_rehearsal_enabled": True}),
        }
    )
    _write_live_feature_audit(settings, state="paper")

    checks = build_llm_live_evidence_checks(
        settings,
        as_of=_AS_OF,
        forecast_evidence=_green_text_forecast(settings),
        profile=ProductionProfile.LLM_LIVE_REHEARSAL,
    )
    assert {check.name: check for check in checks}["llm_live_text_feature_audits_admitted"].passed
    write_llm_live_startup_assertion(
        settings,
        candidate_payload={
            "profile": "llm_live_rehearsal",
            "passed": True,
            "next_allowed_mode": "llm_live_rehearsal",
        },
        as_of=_AS_OF,
    )

    assert_llm_live_startup_allowed(settings, now=_AS_OF + timedelta(minutes=5))


def test_live_llm_rehearsal_requires_absolute_feature_card_dir(tmp_path: Path) -> None:
    settings, _manifest, _card = _live_settings(tmp_path)
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={
                    "live_rehearsal_enabled": True,
                    "text_feature_card_dir": "infra/config/feature_cards/text-live-v1",
                }
            ),
        }
    )

    checks = build_llm_live_evidence_checks(
        settings,
        as_of=_AS_OF,
        forecast_evidence=_green_text_forecast(settings),
        profile=ProductionProfile.LLM_LIVE_REHEARSAL,
    )
    by_name = {check.name: check for check in checks}

    assert by_name["llm_live_text_feature_card_dir_absolute"].passed is False


def test_live_llm_rehearsal_assertion_rejects_true_live_broker(tmp_path: Path) -> None:
    settings, _manifest, _card = _live_settings(tmp_path)
    settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(update={"live_rehearsal_enabled": True}),
        }
    )
    write_llm_live_startup_assertion(
        settings,
        candidate_payload={
            "profile": "llm_live_rehearsal",
            "passed": True,
            "next_allowed_mode": "llm_live_rehearsal",
        },
        as_of=_AS_OF,
    )
    live_broker_settings = settings.model_copy(
        update={
            "broker": BrokerSettings(paper_trading=False),
            "execution": ExecutionSettings(trading_hours_enforced=True),
        }
    )

    with pytest.raises(RuntimeError, match="cannot be used with QP__BROKER__PAPER_TRADING=false"):
        assert_llm_live_startup_allowed(live_broker_settings, now=_AS_OF + timedelta(minutes=5))


def test_live_llm_startup_rejects_stale_assertion(tmp_path: Path) -> None:
    settings, _manifest, _card = _live_settings(tmp_path, as_of=_AS_OF - timedelta(days=2))
    write_llm_live_startup_assertion(
        settings,
        candidate_payload={"passed": True, "next_allowed_mode": "live_ramp_initial"},
        as_of=_AS_OF - timedelta(days=2),
    )

    with pytest.raises(RuntimeError, match="startup assertion is stale"):
        assert_llm_live_startup_allowed(settings, now=_AS_OF)


def test_live_llm_startup_rejects_cap_above_initial_ramp(tmp_path: Path) -> None:
    settings, _manifest, _card = _live_settings(tmp_path, cap=Decimal("0.02"))

    with pytest.raises(RuntimeError, match="LIVE_RAMP_INITIAL <= 0.01"):
        assert_llm_live_startup_allowed(settings, now=_AS_OF)


def test_live_llm_feature_card_hash_mismatch_blocks_candidate(tmp_path: Path) -> None:
    settings, _manifest, card_path = _live_settings(tmp_path)
    _write_live_feature_audit(settings)
    card_path.write_text('{"feature": "live_text_alpha", "changed": true}', encoding="utf-8")

    checks = build_llm_live_evidence_checks(
        settings,
        as_of=_AS_OF,
        forecast_evidence=_green_text_forecast(settings),
    )
    by_name = {check.name: check for check in checks}

    assert by_name["llm_live_text_feature_cards_hash_pinned"].passed is False


def test_live_llm_prediction_schema_hash_mismatch_blocks_candidate(tmp_path: Path) -> None:
    settings, _manifest, _card = _live_settings(tmp_path)
    _write_live_feature_audit(settings)
    evidence = ForecastEvidence(
        source="text",
        model_version="text-v1",
        as_of=_AS_OF,
        horizon="21d",
        observations=25,
        mean_confidence=0.70,
        latest_prediction_at=_AS_OF,
        stale_after=timedelta(hours=24),
        feature_schema_hashes=("wrong",),
    )

    checks = build_llm_live_evidence_checks(settings, as_of=_AS_OF, forecast_evidence=evidence)
    by_name = {check.name: check for check in checks}

    assert by_name["llm_live_prediction_schema_hash_matches"].passed is False


def test_live_llm_prediction_horizon_mismatch_blocks_candidate(tmp_path: Path) -> None:
    settings, _manifest, _card = _live_settings(tmp_path)
    _write_live_feature_audit(settings)
    evidence = _green_text_forecast(settings)
    evidence = ForecastEvidence(
        source=evidence.source,
        model_version=evidence.model_version,
        as_of=evidence.as_of,
        horizon="5d",
        observations=evidence.observations,
        mean_confidence=evidence.mean_confidence,
        latest_prediction_at=evidence.latest_prediction_at,
        stale_after=evidence.stale_after,
        feature_schema_hashes=evidence.feature_schema_hashes,
    )

    checks = build_llm_live_evidence_checks(settings, as_of=_AS_OF, forecast_evidence=evidence)
    by_name = {check.name: check for check in checks}

    assert by_name["llm_live_prediction_source_horizon_21d"].passed is False
