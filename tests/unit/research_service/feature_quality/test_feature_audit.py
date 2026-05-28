from __future__ import annotations

import json
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant_platform.application.features.admission import (
    FeatureAdmissionPolicy,
    assert_feature_artifacts_admitted,
    load_feature_audit_results_from_artifacts,
    ordered_feature_schema_hash,
)
from quant_platform.application.features.governance import (
    FeatureAuditRunRequest,
    FeatureAuditUseCase,
)
from quant_platform.core.domain.research import (
    FeatureDefinition,
    FeatureExpectedSign,
    FeatureProductionState,
    FeatureVector,
)
from quant_platform.infrastructure.repositories.feature_audit_repository import (
    InMemoryFeatureAuditRepository,
)
from quant_platform.infrastructure.support.artifact_store import FileSystemArtifactStore
from quant_platform.services.research_service.feature_quality.audit.feature_audit import (
    FeatureAuditRunner,
    FeatureAuditThresholds,
    load_feature_definition,
)
from quant_platform.services.research_service.feature_quality.audit.gate_evaluators import (
    evaluate_noise_gate,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

_UTC = UTC


def _feature(
    name: str = "quality_alpha",
    *,
    state: FeatureProductionState = FeatureProductionState.SHADOW,
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        version="v1",
        owner="research",
        economic_thesis=(
            "Names with higher audited quality-alpha scores should earn higher "
            "forward returns after controlling for trading costs."
        ),
        source_datasets=("unit-test-synthetic",),
        required_lags=("feature available at or before decision time",),
        valid_universe="unit-test-liquid-equities",
        expected_sign=FeatureExpectedSign.POSITIVE,
        horizon_days=21,
        expected_turnover="low",
        state=state,
        failure_modes=("crowding", "regime flip"),
        risk_exposures=("style",),
    )


def _predictive_samples(days: int = 80, names: int = 10) -> list[SupervisedAlphaSample]:
    start = datetime(2026, 1, 1, tzinfo=_UTC)
    instruments = [uuid.uuid4() for _ in range(names)]
    rows: list[SupervisedAlphaSample] = []
    for day in range(days):
        as_of = start + timedelta(days=day)
        for rank, instrument_id in enumerate(instruments):
            score = -1.0 + 2.0 * rank / max(1, names - 1)
            rows.append(
                SupervisedAlphaSample(
                    as_of=as_of,
                    instrument_id=instrument_id,
                    features={"quality_alpha": score, "baseline": score * 0.1},
                    forward_return=score * 0.01,
                    metadata=(("available_at", as_of.isoformat()),),
                )
            )
    return rows


def _thresholds(**kwargs: object) -> FeatureAuditThresholds:
    defaults = {
        "min_daily_groups": 60,
        "min_coverage": 0.95,
        "min_unique_ratio": 0.01,
        "max_abs_ic": 1.01,
        "min_oos_ic": 0.50,
        "min_icir": 0.0,
        "max_negative_ic_streak": 0,
        "max_turnover": 2.0,
    }
    defaults.update(kwargs)
    return FeatureAuditThresholds(**defaults)


def test_feature_definition_requires_economic_contract() -> None:
    with pytest.raises(ValueError, match="economic_thesis"):
        FeatureDefinition(
            name="x",
            version="v1",
            owner="research",
            economic_thesis="",
            source_datasets=("bars",),
            required_lags=("t-1",),
            valid_universe="us",
            expected_sign=FeatureExpectedSign.POSITIVE,
            horizon_days=21,
            expected_turnover="low",
        )


def test_feature_audit_runner_passes_stable_predictive_feature(tmp_path: Path) -> None:
    manifest = FeatureAuditRunner(
        thresholds=_thresholds(),
        slippage_bps_per_turnover=0.0,
    ).run(
        feature=_feature(),
        samples=_predictive_samples(),
        feature_set_version="features-v1",
        output_root=tmp_path,
    )

    assert manifest.passed is True
    assert manifest.gate_results == {
        "noise": True,
        "leakage": True,
        "ic_stability": True,
        "economic_logic": True,
        "cost": True,
        "incremental": True,
    }
    root = (
        tmp_path / "research" / "feature_audits" / "quality_alpha" / "v1" / str(manifest.audit_id)
    )
    assert (root / "feature_card.json").is_file()
    assert (root / "feature_audit_manifest.json").is_file()


def test_feature_audit_runner_artifact_store_does_not_duplicate_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_root = Path("object-store")
    manifest = FeatureAuditRunner(
        thresholds=_thresholds(),
        slippage_bps_per_turnover=0.0,
        artifact_store=FileSystemArtifactStore(output_root),
    ).run(
        feature=_feature(),
        samples=_predictive_samples(),
        feature_set_version="features-v1",
        output_root=output_root,
    )

    root = (
        tmp_path
        / "object-store"
        / "research"
        / "feature_audits"
        / "quality_alpha"
        / "v1"
        / str(manifest.audit_id)
    )
    assert (root / "feature_audit_manifest.json").is_file()
    assert not (tmp_path / "object-store" / "object-store" / "research").exists()


@pytest.mark.asyncio
async def test_feature_audit_use_case_runs_from_sample_file(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.json"
    rng = random.Random(11)
    samples_path.write_text(
        json.dumps(
            [
                {
                    "as_of": row.as_of.isoformat(),
                    "instrument_id": str(row.instrument_id),
                    "features": dict(row.features),
                    "forward_return": row.forward_return + rng.uniform(-0.004, 0.004),
                    "metadata": list(row.metadata),
                }
                for row in _predictive_samples()
            ]
        ),
        encoding="utf-8",
    )
    card_path = tmp_path / "feature_card.json"
    card_path.write_text(
        json.dumps(
            {
                "name": "quality_alpha",
                "version": "v1",
                "owner": "research",
                "economic_thesis": "Stable audited quality signal earns positive returns.",
                "source_datasets": ["unit-test-synthetic"],
                "required_lags": ["available before decision"],
                "valid_universe": "unit-test",
                "expected_sign": "positive",
                "horizon_days": 21,
                "expected_turnover": "low",
                "state": "paper",
                "failure_modes": ["crowding"],
                "risk_exposures": ["style"],
            }
        ),
        encoding="utf-8",
    )

    result = await FeatureAuditUseCase(object_store_root=tmp_path).run(
        FeatureAuditRunRequest(
            feature_card=card_path,
            samples=samples_path,
            contracts_file=None,
            start=None,
            end=None,
            feature_set_version="features-v1",
            horizon_days=21,
            bar_seconds=86400,
            max_feature_age_days=3,
            output_root=None,
            baseline_features="",
            slippage_bps_per_turnover=0.0,
            min_daily_groups=60,
            min_coverage=0.95,
            min_oos_ic=0.50,
            min_icir=0.0,
            max_negative_ic_streak=0,
            max_turnover=2.0,
            persist=False,
        )
    )

    assert result.payload["feature_name"] == "quality_alpha"
    assert isinstance(result.passed, bool)
    assert Path(str(result.payload["manifest"])).is_file()


def test_feature_audit_runner_blocks_future_available_feature(tmp_path: Path) -> None:
    rows = []
    for sample in _predictive_samples(days=65):
        rows.append(
            SupervisedAlphaSample(
                as_of=sample.as_of,
                instrument_id=sample.instrument_id,
                features=sample.features,
                forward_return=sample.forward_return,
                metadata=(("available_at", (sample.as_of + timedelta(minutes=1)).isoformat()),),
            )
        )

    manifest = FeatureAuditRunner(thresholds=_thresholds()).run(
        feature=_feature(),
        samples=rows,
        feature_set_version="features-v1",
        output_root=tmp_path,
    )

    assert manifest.passed is False
    assert manifest.gate_results["leakage"] is False
    assert any("available_at" in blocker for blocker in manifest.blockers)


def test_feature_audit_runner_blocks_random_noise(tmp_path: Path) -> None:
    rng = random.Random(7)
    rows = _predictive_samples(days=70)
    noisy = [
        SupervisedAlphaSample(
            as_of=row.as_of,
            instrument_id=row.instrument_id,
            features={"quality_alpha": rng.uniform(-1.0, 1.0)},
            forward_return=row.forward_return,
            metadata=row.metadata,
        )
        for row in rows
    ]

    manifest = FeatureAuditRunner(thresholds=_thresholds(min_oos_ic=0.20)).run(
        feature=_feature(),
        samples=noisy,
        feature_set_version="features-v1",
        output_root=tmp_path,
    )

    assert manifest.passed is False
    assert manifest.gate_results["ic_stability"] is False


def test_noise_gate_uses_daily_uniqueness_for_rank_normalized_features() -> None:
    rows = _predictive_samples(days=80, names=15)
    thresholds = _thresholds(min_unique_ratio=0.05)

    report = evaluate_noise_gate(rows, "quality_alpha", thresholds)

    assert report["passed"] is True
    metrics = report["metrics"]
    assert metrics["noise_unique_ratio"] < thresholds.min_unique_ratio
    assert metrics["noise_mean_daily_unique_ratio"] > thresholds.min_unique_ratio
    assert metrics["noise_rank_normalized"] == 1.0


def test_noise_gate_still_blocks_constant_rank_shaped_values() -> None:
    rows = [
        SupervisedAlphaSample(
            as_of=row.as_of,
            instrument_id=row.instrument_id,
            features={"quality_alpha": 0.0},
            forward_return=row.forward_return,
            metadata=row.metadata,
        )
        for row in _predictive_samples(days=80, names=15)
    ]

    report = evaluate_noise_gate(rows, "quality_alpha", _thresholds(min_unique_ratio=0.05))

    assert report["passed"] is False
    assert report["metrics"]["noise_rank_normalized"] == 0.0
    assert any("unique_ratio" in blocker for blocker in report["blockers"])


def test_noise_gate_blocks_unnormalized_alpha_feature() -> None:
    """Raw-scale alpha features (values outside [-1, 1]) are blocked (WS2)."""
    rows = [
        SupervisedAlphaSample(
            as_of=row.as_of,
            instrument_id=row.instrument_id,
            features={"quality_alpha": row.features["quality_alpha"] * 100.0},
            forward_return=row.forward_return,
            metadata=row.metadata,
        )
        for row in _predictive_samples(days=80, names=15)
    ]

    blocked = evaluate_noise_gate(rows, "quality_alpha", _thresholds())
    assert blocked["passed"] is False
    assert any("rank-normalized" in blocker for blocker in blocked["blockers"])

    # The requirement is configurable: disabling it drops the rank blocker.
    allowed = evaluate_noise_gate(rows, "quality_alpha", _thresholds(require_rank_normalized=False))
    assert all("rank-normalized" not in blocker for blocker in allowed["blockers"])


def test_feature_audit_runner_blocks_cost_heavy_turnover(tmp_path: Path) -> None:
    rows = []
    instruments = [uuid.uuid4() for _ in range(4)]
    start = datetime(2026, 1, 1, tzinfo=_UTC)
    for day in range(70):
        sign = 1.0 if day % 2 == 0 else -1.0
        as_of = start + timedelta(days=day)
        for idx, instrument_id in enumerate(instruments):
            score = sign if idx < 2 else -sign
            rows.append(
                SupervisedAlphaSample(
                    as_of=as_of,
                    instrument_id=instrument_id,
                    features={"quality_alpha": score},
                    forward_return=score * 0.01,
                    metadata=(("available_at", as_of.isoformat()),),
                )
            )

    manifest = FeatureAuditRunner(
        thresholds=_thresholds(max_turnover=0.50),
        slippage_bps_per_turnover=1000.0,
    ).run(
        feature=_feature(),
        samples=rows,
        feature_set_version="features-v1",
        output_root=tmp_path,
    )

    assert manifest.passed is False
    assert manifest.gate_results["cost"] is False


def test_feature_audit_runner_blocks_redundant_incremental_feature(tmp_path: Path) -> None:
    manifest = FeatureAuditRunner(
        thresholds=_thresholds(max_baseline_correlation=0.80),
        slippage_bps_per_turnover=0.0,
        baseline_features=("baseline",),
    ).run(
        feature=_feature("quality_alpha"),
        samples=[
            SupervisedAlphaSample(
                as_of=row.as_of,
                instrument_id=row.instrument_id,
                features={
                    "quality_alpha": row.features["quality_alpha"],
                    "baseline": row.features["quality_alpha"],
                },
                forward_return=row.forward_return,
                metadata=row.metadata,
            )
            for row in _predictive_samples(days=70)
        ],
        feature_set_version="features-v1",
        output_root=tmp_path,
    )

    assert manifest.passed is False
    assert manifest.gate_results["incremental"] is False


@pytest.mark.asyncio
async def test_feature_audit_repository_round_trips_latest(tmp_path: Path) -> None:
    manifest = FeatureAuditRunner(
        thresholds=_thresholds(),
        slippage_bps_per_turnover=0.0,
    ).run(
        feature=_feature(),
        samples=_predictive_samples(),
        feature_set_version="features-v1",
        output_root=tmp_path,
    )
    result = manifest.to_result("artifact://feature_audit_manifest.json")
    repo = InMemoryFeatureAuditRepository()

    await repo.save_feature_audit(result)
    latest = await repo.latest_feature_audit("quality_alpha", "v1")

    assert latest == result


def test_feature_admission_policy_accepts_audited_paper_feature(tmp_path: Path) -> None:
    manifest = FeatureAuditRunner(
        thresholds=_thresholds(),
        slippage_bps_per_turnover=0.0,
    ).run(
        feature=_feature(state=FeatureProductionState.PAPER),
        samples=_predictive_samples(),
        feature_set_version="features-v1",
        output_root=tmp_path,
    )
    policy = FeatureAdmissionPolicy([manifest.to_result("artifact://audit.json")])

    decision = policy.evaluate(
        feature_names=("quality_alpha",),
        feature_versions={"quality_alpha": "v1"},
        feature_set_version="features-v1",
        minimum_state=FeatureProductionState.PAPER,
        model_feature_schema_hash=ordered_feature_schema_hash(("quality_alpha",)),
    )

    assert decision.passed is True
    assert decision.blockers == ()


def test_feature_admission_policy_fails_closed_on_schema_mismatch(tmp_path: Path) -> None:
    manifest = FeatureAuditRunner(
        thresholds=_thresholds(),
        slippage_bps_per_turnover=0.0,
    ).run(
        feature=_feature(state=FeatureProductionState.PAPER),
        samples=_predictive_samples(),
        feature_set_version="features-v1",
        output_root=tmp_path,
    )
    policy = FeatureAdmissionPolicy([manifest.to_result("artifact://audit.json")])

    with pytest.raises(RuntimeError, match="feature_schema_hash"):
        policy.assert_admitted(
            feature_names=("quality_alpha",),
            feature_versions={"quality_alpha": "v1"},
            feature_set_version="features-v1",
            minimum_state=FeatureProductionState.PAPER,
            model_feature_schema_hash="wrong",
        )


def test_feature_admission_policy_validates_runtime_vectors(tmp_path: Path) -> None:
    manifest = FeatureAuditRunner(
        thresholds=_thresholds(),
        slippage_bps_per_turnover=0.0,
    ).run(
        feature=_feature(state=FeatureProductionState.LIVE),
        samples=_predictive_samples(),
        feature_set_version="features-v1",
        output_root=tmp_path,
    )
    vector = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        as_of=datetime(2026, 5, 1, tzinfo=_UTC),
        feature_set_version="features-v1",
        features={"quality_alpha": 1.0},
        strategy_run_id=uuid.uuid4(),
    )
    policy = FeatureAdmissionPolicy([manifest.to_result("artifact://audit.json")])

    policy.assert_vectors_admitted(
        vectors=(vector,),
        feature_versions={"quality_alpha": "v1"},
        minimum_state=FeatureProductionState.LIVE,
        model_feature_schema_hash=ordered_feature_schema_hash(("quality_alpha",)),
    )


def test_feature_admission_loads_local_audit_artifacts(tmp_path: Path) -> None:
    manifest = FeatureAuditRunner(
        thresholds=_thresholds(),
        slippage_bps_per_turnover=0.0,
    ).run(
        feature=_feature(state=FeatureProductionState.PAPER),
        samples=_predictive_samples(),
        feature_set_version="features-v1",
        output_root=tmp_path,
    )

    audits = load_feature_audit_results_from_artifacts(
        tmp_path,
        feature_names=("quality_alpha",),
    )
    decision = assert_feature_artifacts_admitted(
        audit_root=tmp_path,
        feature_names=("quality_alpha",),
        feature_versions={"quality_alpha": "v1"},
        feature_set_version="features-v1",
        minimum_state=FeatureProductionState.PAPER,
        model_feature_schema_hash=ordered_feature_schema_hash(("quality_alpha",)),
    )

    assert audits[0].audit_id == manifest.audit_id
    assert decision.passed is True


def test_feature_artifact_admission_fails_closed_when_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no feature audit manifests"):
        assert_feature_artifacts_admitted(
            audit_root=tmp_path,
            feature_names=("quality_alpha",),
            feature_versions={"quality_alpha": "v1"},
            feature_set_version="features-v1",
            minimum_state=FeatureProductionState.PAPER,
            model_feature_schema_hash=ordered_feature_schema_hash(("quality_alpha",)),
        )


def test_load_feature_definition_from_json(tmp_path: Path) -> None:
    path = tmp_path / "feature.json"
    path.write_text(
        json.dumps(
            {
                "name": "quality_alpha",
                "version": "v1",
                "owner": "research",
                "economic_thesis": "Quality alpha should predict forward returns in liquid equities.",
                "source_datasets": ["bars"],
                "required_lags": ["t-1"],
                "valid_universe": "us-liquid",
                "expected_sign": "positive",
                "horizon_days": 21,
                "expected_turnover": "low",
                "state": "shadow",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_feature_definition(path)

    assert loaded.name == "quality_alpha"
    assert loaded.state == FeatureProductionState.SHADOW
