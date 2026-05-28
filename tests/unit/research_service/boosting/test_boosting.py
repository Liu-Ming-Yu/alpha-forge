"""Tests for XGBoost boosted-tree training and scoring."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

import quant_platform.services.research_service.boosting as boosting
from quant_platform.core.domain.research import FeatureVector, RunStatus, RunType, StrategyRun
from quant_platform.core.domain.signals import SignalScore
from quant_platform.infrastructure.performance import InMemoryPerformanceRepository

if TYPE_CHECKING:
    from pathlib import Path


def _strategy_run() -> StrategyRun:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="xsec",
        strategy_version="boost-v1",
        run_type=RunType.BACKTEST,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=now,
    )


def _samples() -> list[boosting.BoostingSample]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    out: list[boosting.BoostingSample] = []
    for day in range(6):
        as_of = start + timedelta(days=day)
        for rank in range(4):
            out.append(
                boosting.BoostingSample(
                    as_of=as_of,
                    instrument_id=uuid.uuid4(),
                    features={"alpha": float(rank), "beta": float(day), "_vol": 0.2},
                    forward_return=float(rank) / 100.0,
                )
            )
    return out


class _FakeBoostingModel:
    model_version = "xgb-fake"
    feature_set_version = "1.0.0"
    feature_schema_hash = "schema-hash"
    device = "cpu"

    def feature_coverage(self, features: dict[str, float]) -> float:
        return 1.0 if "alpha" in features else 0.0

    def score(
        self,
        vectors: list[FeatureVector],
        strategy_run: StrategyRun,
    ) -> list[SignalScore]:
        return [
            SignalScore(
                score_id=uuid.uuid4(),
                instrument_id=vector.instrument_id,
                strategy_run_id=strategy_run.run_id,
                as_of=vector.as_of,
                score=float(vector.features["alpha"]),
                confidence=0.8,
                model_version=self.model_version,
                feature_vector_id=vector.vector_id,
            )
            for vector in vectors
        ]


def test_feature_schema_hash_is_order_sensitive() -> None:
    assert boosting.feature_schema_hash(["a", "b"]) == boosting.feature_schema_hash(["a", "b"])
    assert boosting.feature_schema_hash(["a", "b"]) != boosting.feature_schema_hash(["b", "a"])


def test_resolve_device_falls_back_to_cpu_when_auto_and_wsl_gpu_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        boosting,
        "_nvidia_smi_status",
        lambda: {"available": False, "detail": "GPU access blocked by the operating system"},
    )
    assert boosting.resolve_xgboost_device("auto", xgb=SimpleNamespace()) == "cpu"
    with pytest.raises(RuntimeError, match="required"):
        boosting.resolve_xgboost_device("auto", require_gpu=True, xgb=SimpleNamespace())


def test_train_writes_manifest_and_model_scores_with_cpu(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    manifest, manifest_path = boosting.train_xgboost_ranker(
        _samples(),
        boosting.BoostingTrainConfig(
            model_version="xgb-test",
            feature_set_version="1.0.0",
            output_root=tmp_path,
            device="cpu",
            purge_days=0,
            num_boost_round=5,
            early_stopping_rounds=2,
        ),
    )

    assert manifest_path.is_file()
    assert (tmp_path / "xgb-test" / "model.json").is_file()
    assert (tmp_path / "xgb-test" / "metrics.json").is_file()
    assert manifest.feature_names == ["alpha", "beta"]
    assert manifest.feature_versions == {"alpha": "1.0.0", "beta": "1.0.0"}
    assert manifest.feature_schema_hash == boosting.feature_schema_hash(["alpha", "beta"])

    model = boosting.XGBoostRankSignalModel(manifest_path, device="cpu")
    run = _strategy_run()
    vectors = [
        FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=uuid.uuid4(),
            as_of=run.created_at,
            feature_set_version="1.0.0",
            features={"alpha": float(i), "beta": 1.0, "extra": 99.0},
            strategy_run_id=run.run_id,
        )
        for i in range(3)
    ]
    scores = model.score(vectors, run)

    assert len(scores) == 3
    assert all(-1.0 <= score.score <= 1.0 for score in scores)
    assert {score.model_version for score in scores} == {"xgb-test"}


def test_model_rejects_missing_nonfinite_and_wrong_feature_version(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    _, manifest_path = boosting.train_xgboost_ranker(
        _samples(),
        boosting.BoostingTrainConfig(
            model_version="xgb-validate",
            feature_set_version="features-v1",
            output_root=tmp_path,
            device="cpu",
            purge_days=0,
            num_boost_round=3,
            early_stopping_rounds=1,
        ),
    )
    model = boosting.XGBoostRankSignalModel(manifest_path, device="cpu")
    run = _strategy_run()

    base = dict(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        as_of=run.created_at,
        strategy_run_id=run.run_id,
    )
    with pytest.raises(ValueError, match="missing"):
        model.score(
            [
                FeatureVector(
                    **base,
                    feature_set_version="features-v1",
                    features={"alpha": 1.0},
                )
            ],
            run,
        )
    with pytest.raises(ValueError, match="not finite"):
        model.score(
            [
                FeatureVector(
                    **base,
                    feature_set_version="features-v1",
                    features={"alpha": 1.0, "beta": float("nan")},
                )
            ],
            run,
        )
    with pytest.raises(ValueError, match="feature_set_version mismatch"):
        model.score(
            [
                FeatureVector(
                    **base,
                    feature_set_version="other",
                    features={"alpha": 1.0, "beta": 1.0},
                )
            ],
            run,
        )


@pytest.mark.asyncio
async def test_shadow_boosting_scorer_writes_jsonl(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    _, manifest_path = boosting.train_xgboost_ranker(
        _samples(),
        boosting.BoostingTrainConfig(
            model_version="xgb-shadow",
            feature_set_version="1.0.0",
            output_root=tmp_path / "models",
            device="cpu",
            purge_days=0,
            num_boost_round=3,
            early_stopping_rounds=1,
        ),
    )
    model = boosting.XGBoostRankSignalModel(manifest_path, device="cpu")
    scorer = boosting.ShadowBoostingScorer(model=model, artifact_root=tmp_path / "shadow")
    run = _strategy_run()
    instrument_id = uuid.uuid4()

    path = await scorer.score_cycle(
        feature_data={instrument_id: {"alpha": 1.0, "beta": 2.0}},
        primary_scores=[],
        strategy_run=run,
        as_of=run.created_at,
    )

    assert path is not None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["instrument_id"] == str(instrument_id)
    assert rows[0]["model_version"] == "xgb-shadow"
    assert rows[0]["feature_coverage"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_shadow_boosting_scorer_persists_prediction_evidence(tmp_path: Path) -> None:
    repo = InMemoryPerformanceRepository()
    scorer = boosting.ShadowBoostingScorer(
        model=_FakeBoostingModel(),
        artifact_root=tmp_path / "shadow",
        prediction_evidence_repo=repo,
        horizon="21d",
    )
    run = _strategy_run()
    instrument_id = uuid.uuid4()

    await scorer.score_cycle(
        feature_data={instrument_id: {"alpha": 0.75}},
        primary_scores=[],
        strategy_run=run,
        as_of=run.created_at,
    )

    rows = await repo.list_prediction_results(
        source="xgboost",
        model_version="xgb-fake",
        strategy_run_id=run.run_id,
    )
    assert len(rows) == 1
    assert rows[0].instrument_id == instrument_id
    assert rows[0].rank_score == pytest.approx(0.75)
    assert rows[0].confidence == pytest.approx(0.8)
    assert rows[0].metadata["feature_set_version"] == "1.0.0"


# ──────────────────────────── GPU tests ────────────────────────────


@pytest.fixture(scope="module")
def _require_gpu() -> None:
    """Skip the test if no CUDA GPU is reachable via XGBoost smoke test."""
    pytest.importorskip("xgboost")
    result = boosting.gpu_check()
    smoke = result.get("cuda_smoke", {})
    if not smoke.get("ok"):  # type: ignore[union-attr]
        pytest.skip(f"No CUDA GPU available: {smoke.get('detail')}")


def test_gpu_check_returns_expected_structure() -> None:
    """gpu_check() always runs and returns a well-formed diagnostic dict."""
    pytest.importorskip("xgboost")
    result = boosting.gpu_check()
    assert "nvidia_smi" in result
    assert "xgboost" in result
    assert "cuda_smoke" in result
    smoke = result["cuda_smoke"]
    assert isinstance(smoke, dict)
    assert "ok" in smoke  # type: ignore[operator]
    assert "detail" in smoke  # type: ignore[operator]
    assert isinstance(smoke["ok"], bool)  # type: ignore[index]


def test_cuda_smoke_never_raises() -> None:
    """_cuda_smoke always returns (bool, str) — never raises regardless of GPU state."""
    xgb = pytest.importorskip("xgboost")
    ok, detail = boosting._cuda_smoke(xgb)
    assert isinstance(ok, bool)
    assert isinstance(detail, str)


def test_resolve_device_returns_cuda_when_smoke_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_xgboost_device returns 'cuda' when nvidia-smi is available and smoke succeeds."""
    monkeypatch.setattr(boosting, "_nvidia_smi_status", lambda: {"available": True, "detail": "OK"})
    monkeypatch.setattr(boosting, "_cuda_smoke", lambda _xgb: (True, "cuda smoke train completed"))
    assert boosting.resolve_xgboost_device("cuda", xgb=SimpleNamespace()) == "cuda"
    assert boosting.resolve_xgboost_device("auto", xgb=SimpleNamespace()) == "cuda"


def test_resolve_device_raises_for_explicit_cuda_when_smoke_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_xgboost_device raises RuntimeError when device='cuda' but smoke fails."""
    monkeypatch.setattr(boosting, "_nvidia_smi_status", lambda: {"available": True, "detail": "OK"})
    monkeypatch.setattr(boosting, "_cuda_smoke", lambda _xgb: (False, "No visible GPU"))
    with pytest.raises(RuntimeError, match="unavailable"):
        boosting.resolve_xgboost_device("cuda", xgb=SimpleNamespace())


@pytest.mark.gpu
def test_train_and_score_with_cuda(tmp_path: Path, _require_gpu: None) -> None:
    """Full train → manifest → score cycle using device='cuda'."""
    manifest, manifest_path = boosting.train_xgboost_ranker(
        _samples(),
        boosting.BoostingTrainConfig(
            model_version="xgb-gpu",
            feature_set_version="1.0.0",
            output_root=tmp_path,
            device="cuda",
            purge_days=0,
            num_boost_round=5,
            early_stopping_rounds=2,
        ),
    )
    assert manifest.device == "cuda"
    assert manifest_path.is_file()
    assert (tmp_path / "xgb-gpu" / "model.json").is_file()
    assert manifest.feature_names == ["alpha", "beta"]

    model = boosting.XGBoostRankSignalModel(manifest_path, device="cuda")
    run = _strategy_run()
    vectors = [
        FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=uuid.uuid4(),
            as_of=run.created_at,
            feature_set_version="1.0.0",
            features={"alpha": float(i), "beta": 1.0, "extra": 99.0},
            strategy_run_id=run.run_id,
        )
        for i in range(3)
    ]
    scores = model.score(vectors, run)
    assert len(scores) == 3
    assert all(-1.0 <= s.score <= 1.0 for s in scores)
    assert {s.model_version for s in scores} == {"xgb-gpu"}


@pytest.mark.gpu
async def test_shadow_scorer_with_cuda(tmp_path: Path, _require_gpu: None) -> None:
    """ShadowBoostingScorer writes correct JSONL when model runs on CUDA."""
    _, manifest_path = boosting.train_xgboost_ranker(
        _samples(),
        boosting.BoostingTrainConfig(
            model_version="xgb-gpu-shadow",
            feature_set_version="1.0.0",
            output_root=tmp_path / "models",
            device="cuda",
            purge_days=0,
            num_boost_round=3,
            early_stopping_rounds=1,
        ),
    )
    model = boosting.XGBoostRankSignalModel(manifest_path, device="cuda")
    scorer = boosting.ShadowBoostingScorer(model=model, artifact_root=tmp_path / "shadow")
    run = _strategy_run()
    instrument_id = uuid.uuid4()

    path = await scorer.score_cycle(
        feature_data={instrument_id: {"alpha": 1.0, "beta": 2.0}},
        primary_scores=[],
        strategy_run=run,
        as_of=run.created_at,
    )

    assert path is not None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["instrument_id"] == str(instrument_id)
    assert rows[0]["model_version"] == "xgb-gpu-shadow"
