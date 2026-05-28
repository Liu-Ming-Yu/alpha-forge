from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from quant_platform.bootstrap.alpha_forecast_ops import materialize as forecast_materialize
from quant_platform.bootstrap.alpha_forecast_ops import materialize_alpha_forecasts
from quant_platform.bootstrap.alpha_forecast_ops import sources as forecast_sources
from quant_platform.config import (
    AlphaSettings,
    BoostingSettings,
    LLMSettings,
    PlatformSettings,
    StorageSettings,
)
from quant_platform.core.domain.research import FeatureVector
from quant_platform.core.domain.signals import SignalScore
from quant_platform.infrastructure.performance import InMemoryPerformanceRepository
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository


@pytest.mark.asyncio
async def test_materialize_forecasts_writes_deterministic_promoted_source_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    as_of = datetime(2026, 5, 14, tzinfo=UTC)
    contracts = {
        uuid.uuid5(uuid.NAMESPACE_URL, "aapl"): {"symbol": "AAPL"},
        uuid.uuid5(uuid.NAMESPACE_URL, "msft"): {"symbol": "MSFT"},
    }
    contracts_file = tmp_path / "contracts.json"
    contracts_file.write_text(
        json.dumps({str(key): value for key, value in contracts.items()}),
        encoding="utf-8",
    )
    feature_repo = InMemoryFeatureRepository()
    await _store_source_vectors(feature_repo, contracts, as_of)
    performance_repo = InMemoryPerformanceRepository()
    monkeypatch.setattr(
        forecast_materialize,
        "build_performance_repository",
        lambda _dsn: performance_repo,
    )
    monkeypatch.setattr(forecast_sources, "_load_xgboost_model", lambda *_: _FakeXGBoost())

    settings = _settings()
    payload = await materialize_alpha_forecasts(
        settings,
        contracts_file=contracts_file,
        as_of=as_of,
        sources=("text", "event", "intraday", "xgboost"),
        horizon="21d",
        xgboost_manifest=tmp_path / "manifest.json",
        fail_on_missing=True,
        feature_repo=feature_repo,
    )
    second_payload = await materialize_alpha_forecasts(
        settings,
        contracts_file=contracts_file,
        as_of=as_of,
        sources=("text", "event", "intraday", "xgboost"),
        horizon="21d",
        xgboost_manifest=tmp_path / "manifest.json",
        fail_on_missing=True,
        feature_repo=feature_repo,
    )

    rows = await performance_repo.list_prediction_results(limit=20)
    assert payload["passed"] is True
    assert payload["prediction_results_saved"] == 8
    assert second_payload["prediction_results_saved"] == 8
    assert len(rows) == 8
    assert {row.source for row in rows} == {"event", "intraday", "text", "xgboost"}
    assert all(row.as_of == as_of for row in rows)
    assert all(row.horizon == "21d" for row in rows)
    assert all(row.confidence == 1.0 for row in rows)
    assert {row.metadata["feature_set_version"] for row in rows if row.source != "xgboost"} == {
        "event-set",
        "intraday-set",
        "text-set",
    }


@pytest.mark.asyncio
async def test_materialize_forecasts_fails_closed_without_partial_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    as_of = datetime(2026, 5, 14, tzinfo=UTC)
    instrument_id = uuid.uuid5(uuid.NAMESPACE_URL, "aapl")
    contracts_file = tmp_path / "contracts.json"
    contracts_file.write_text(
        json.dumps({str(instrument_id): {"symbol": "AAPL"}}),
        encoding="utf-8",
    )
    feature_repo = InMemoryFeatureRepository()
    await feature_repo.store_vector(
        _vector(
            instrument_id=instrument_id,
            as_of=as_of,
            feature_set_version="text-set",
            features={"text_alpha": 0.3},
        )
    )
    performance_repo = InMemoryPerformanceRepository()
    monkeypatch.setattr(
        forecast_materialize,
        "build_performance_repository",
        lambda _dsn: performance_repo,
    )

    payload = await materialize_alpha_forecasts(
        _settings(),
        contracts_file=contracts_file,
        as_of=as_of,
        sources=("text", "event"),
        horizon="21d",
        xgboost_manifest=None,
        fail_on_missing=True,
        feature_repo=feature_repo,
    )

    assert payload["passed"] is False
    assert payload["prediction_results_saved"] == 0
    assert any("event missing exact-as-of vectors" in blocker for blocker in payload["blockers"])
    assert await performance_repo.list_prediction_results(limit=20) == []


@pytest.mark.asyncio
async def test_materialize_forecasts_requires_manifest_for_xgboost(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    as_of = datetime(2026, 5, 14, tzinfo=UTC)
    instrument_id = uuid.uuid5(uuid.NAMESPACE_URL, "aapl")
    contracts_file = tmp_path / "contracts.json"
    contracts_file.write_text(
        json.dumps({str(instrument_id): {"symbol": "AAPL"}}),
        encoding="utf-8",
    )
    performance_repo = InMemoryPerformanceRepository()
    monkeypatch.setattr(
        forecast_materialize,
        "build_performance_repository",
        lambda _dsn: performance_repo,
    )

    payload = await materialize_alpha_forecasts(
        _settings(),
        contracts_file=contracts_file,
        as_of=as_of,
        sources=("xgboost",),
        horizon="21d",
        xgboost_manifest=None,
        fail_on_missing=True,
        feature_repo=InMemoryFeatureRepository(),
    )

    assert payload["passed"] is False
    assert payload["prediction_results_saved"] == 0
    assert payload["blockers"] == ["xgboost forecast materialization requires --xgboost-manifest"]


@pytest.mark.asyncio
async def test_materialize_forecasts_seeds_live_llm_rehearsal_before_startup_assertion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    as_of = datetime(2026, 5, 14, tzinfo=UTC)
    instrument_id = uuid.uuid5(uuid.NAMESPACE_URL, "aapl")
    contracts_file = tmp_path / "contracts.json"
    contracts_file.write_text(
        json.dumps({str(instrument_id): {"symbol": "AAPL"}}),
        encoding="utf-8",
    )
    feature_repo = InMemoryFeatureRepository()
    await feature_repo.store_vector(
        _vector(
            instrument_id=instrument_id,
            as_of=as_of,
            feature_set_version="text-live-v1",
            features={"text_alpha": 0.8},
        )
    )
    performance_repo = InMemoryPerformanceRepository()
    monkeypatch.setattr(
        forecast_materialize,
        "build_performance_repository",
        lambda _dsn: performance_repo,
    )

    payload = await materialize_alpha_forecasts(
        PlatformSettings(
            _env_file=None,
            storage=StorageSettings(object_store_root=str(tmp_path), postgres_dsn=""),
            alpha=AlphaSettings(
                ensemble_mode="live",
                source_weights={"classical": 0.99, "text": 0.01},
                max_non_classical_weight=0.01,
            ),
            llm=LLMSettings(
                live_mode_enabled=True,
                live_rehearsal_enabled=True,
                text_feature_set_version="text-live-v1",
                text_feature_weights={"text_alpha": 1.0},
            ),
        ),
        contracts_file=contracts_file,
        as_of=as_of,
        sources=("text",),
        horizon="21d",
        xgboost_manifest=None,
        fail_on_missing=True,
        feature_repo=feature_repo,
    )

    rows = await performance_repo.list_prediction_results(source="text")
    assert payload["passed"] is True
    assert payload["prediction_results_saved"] == 1
    assert len(rows) == 1
    assert rows[0].source == "text"


async def _store_source_vectors(
    repo: InMemoryFeatureRepository,
    contracts: dict[uuid.UUID, dict[str, object]],
    as_of: datetime,
) -> None:
    for instrument_id in contracts:
        await repo.store_vector(
            _vector(
                instrument_id=instrument_id,
                as_of=as_of,
                feature_set_version="text-set",
                features={"text_alpha": 0.3},
            )
        )
        await repo.store_vector(
            _vector(
                instrument_id=instrument_id,
                as_of=as_of,
                feature_set_version="event-set",
                features={"event_alpha": -0.2},
            )
        )
        await repo.store_vector(
            _vector(
                instrument_id=instrument_id,
                as_of=as_of,
                feature_set_version="intraday-set",
                features={"intraday_alpha": 0.1},
            )
        )
        await repo.store_vector(
            _vector(
                instrument_id=instrument_id,
                as_of=as_of,
                feature_set_version="paper-alpha-composite-v1",
                features={"xgb_alpha": 0.4},
            )
        )


def _settings() -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn=""),
        llm=LLMSettings(
            text_feature_set_version="text-set",
            text_feature_weights={"text_alpha": 1.0},
        ),
        alpha=AlphaSettings(
            source_weights={
                "classical": 0.70,
                "xgboost": 0.15,
                "text": 0.05,
                "event": 0.05,
                "intraday": 0.05,
            },
            event_feature_set_version="event-set",
            event_feature_weights={"event_alpha": 1.0},
            intraday_feature_set_version="intraday-set",
            intraday_feature_weights={"intraday_alpha": 1.0},
        ),
        boosting=BoostingSettings(artifact_manifest=""),
    )


def _vector(
    *,
    instrument_id: uuid.UUID,
    as_of: datetime,
    feature_set_version: str,
    features: dict[str, float],
) -> FeatureVector:
    return FeatureVector(
        vector_id=uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{instrument_id}:{feature_set_version}:{as_of.isoformat()}",
        ),
        instrument_id=instrument_id,
        as_of=as_of,
        feature_set_version=feature_set_version,
        features=features,
        strategy_run_id=uuid.uuid4(),
        artifact_uri="file:///tmp/features.parquet",
        available_at=as_of,
    )


class _FakeXGBoost:
    model_version = "xgb-v1"
    feature_set_version = "paper-alpha-composite-v1"
    feature_names = ["xgb_alpha"]
    feature_schema_hash = "xgb-schema"
    device = "cpu"

    def feature_coverage(self, features: dict[str, float]) -> float:
        return 1.0 if "xgb_alpha" in features else 0.0

    def score(self, vectors: list[FeatureVector], strategy_run: Any) -> list[SignalScore]:
        return [
            SignalScore(
                score_id=uuid.uuid4(),
                instrument_id=vector.instrument_id,
                strategy_run_id=strategy_run.run_id,
                as_of=vector.as_of,
                score=float(vector.features["xgb_alpha"]),
                confidence=1.0,
                model_version=self.model_version,
                feature_vector_id=vector.vector_id,
            )
            for vector in vectors
        ]
