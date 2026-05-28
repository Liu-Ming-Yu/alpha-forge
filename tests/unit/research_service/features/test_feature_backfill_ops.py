"""Tests for the bootstrap feature-backfill operation wiring.

These exercise ``research.features.backfill_ops`` (durable-input
guard, session assembly, result payload) with a patched paper session and a
stub feature-set backfiller so no live Postgres or feature computation is
required.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.research import FeaturesBackfillRequest
from quant_platform.config import PlatformSettings, StorageSettings
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.research import FeatureVector
from quant_platform.research.features import backfill_ops


class _Repo:
    def __init__(self) -> None:
        self.vectors: list[FeatureVector] = []

    async def store_vector(self, vector: FeatureVector) -> None:
        self.vectors.append(vector)

    async def get_vectors(
        self,
        instrument_ids: list[uuid.UUID],
        feature_set_version: str,
        as_of: datetime,
    ) -> list[FeatureVector]:
        wanted = set(instrument_ids)
        return [
            vector
            for vector in self.vectors
            if vector.instrument_id in wanted
            and vector.feature_set_version == feature_set_version
            and vector.as_of <= as_of
        ]


class _BarStore:
    def __init__(self, bars_by_instrument: dict[uuid.UUID, list[MarketBar]]) -> None:
        self._bars_by_instrument = bars_by_instrument

    async def get_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        return [
            bar
            for bar in self._bars_by_instrument.get(instrument_id, [])
            if start <= bar.timestamp <= end
        ]


def _bars(instrument_id: uuid.UUID, count: int, *, end: datetime) -> list[MarketBar]:
    rows: list[MarketBar] = []
    for offset in range(count):
        price = Decimal("100") + Decimal(offset)
        rows.append(
            MarketBar(
                bar_id=uuid.uuid4(),
                instrument_id=instrument_id,
                timestamp=end - timedelta(days=count - 1 - offset),
                bar_seconds=86400,
                open=price - Decimal("0.50"),
                high=price + Decimal("1.00"),
                low=price - Decimal("1.00"),
                close=price,
                volume=1_000_000 + offset * 1_000,
            )
        )
    return rows


def _settings(tmp_path, *, postgres: bool = True) -> PlatformSettings:
    root = tmp_path / "objects"
    root.mkdir(exist_ok=True)
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(
            postgres_dsn="postgresql+psycopg://u:p@localhost/db" if postgres else "",
            object_store_root=str(root),
        ),
    )


def _contracts_file(tmp_path) -> str:
    contracts = tmp_path / "contracts.json"
    contracts.write_text(json.dumps({str(uuid.uuid4()): {"symbol": "AAPL"}}), encoding="utf-8")
    return str(contracts)


def _request(tmp_path, **overrides) -> FeaturesBackfillRequest:
    values: dict[str, object] = {
        "command": "backfill",
        "contracts_file": _contracts_file(tmp_path),
        "start": datetime(2026, 1, 22, tzinfo=UTC),
        "end": datetime(2026, 1, 22, tzinfo=UTC),
        "feature_set_version": "1.0.0",
        "bar_seconds": 86400,
        "lookback_days": 30,
        "dry_run": False,
    }
    values.update(overrides)
    return FeaturesBackfillRequest(**values)  # type: ignore[arg-type]


def _patch_session(monkeypatch, *, repo: _Repo, bar_store: _BarStore, instrument_id: uuid.UUID):
    async def _verify(_settings) -> None:
        return None

    def _session(**_kwargs):
        return SimpleNamespace(
            contract_master=SimpleNamespace(
                list_active=lambda: [SimpleNamespace(instrument_id=instrument_id)]
            ),
            bar_store=bar_store,
            feature_repo=repo,
        )

    monkeypatch.setattr(backfill_ops, "_verify_postgres_schema_if_configured", _verify)
    monkeypatch.setattr(backfill_ops, "create_paper_session", _session)


def _stub_backfiller(monkeypatch):
    async def _backfiller(
        *,
        bar_history,
        feature_repo,
        strategy_run_id,
        as_of,
        feature_set_version,
        artifact_uri,
        dry_run,
        events_by_instrument=None,
    ) -> int:
        if not dry_run:
            for instrument_id in bar_history:
                await feature_repo.store_vector(
                    FeatureVector(
                        vector_id=uuid.uuid4(),
                        instrument_id=instrument_id,
                        strategy_run_id=strategy_run_id,
                        as_of=as_of,
                        available_at=as_of,
                        feature_set_version=feature_set_version,
                        features={"stub_feature": 1.0},
                    )
                )
        return len(bar_history)

    monkeypatch.setattr(backfill_ops, "backfill_ohlcv_feature_set", _backfiller)


@pytest.mark.asyncio
async def test_features_backfill_requires_durable_postgres(tmp_path) -> None:
    with pytest.raises(OperatorUsageError, match="durable historical feature vectors"):
        await backfill_ops._features_backfill(
            _settings(tmp_path, postgres=False),
            _request(tmp_path),
        )


@pytest.mark.asyncio
async def test_features_backfill_rejects_short_lookback(tmp_path) -> None:
    with pytest.raises(OperatorUsageError, match="lookback-days"):
        await backfill_ops._features_backfill(
            _settings(tmp_path),
            _request(tmp_path, lookback_days=5),
        )


@pytest.mark.asyncio
async def test_features_backfill_stores_vectors_through_session(tmp_path, monkeypatch) -> None:
    instrument_id = uuid.uuid4()
    end = datetime(2026, 1, 22, tzinfo=UTC)
    repo = _Repo()
    bar_store = _BarStore({instrument_id: _bars(instrument_id, 60, end=end)})
    _patch_session(monkeypatch, repo=repo, bar_store=bar_store, instrument_id=instrument_id)
    _stub_backfiller(monkeypatch)

    result = await backfill_ops._features_backfill(
        _settings(tmp_path),
        _request(tmp_path),
    )

    assert result.payload["vectors_stored"] == 1
    assert result.payload["feature_set_version"] == "1.0.0"
    assert len(repo.vectors) == 1


@pytest.mark.asyncio
async def test_features_backfill_dry_run_reports_would_store(tmp_path, monkeypatch) -> None:
    instrument_id = uuid.uuid4()
    end = datetime(2026, 1, 22, tzinfo=UTC)
    repo = _Repo()
    bar_store = _BarStore({instrument_id: _bars(instrument_id, 60, end=end)})
    _patch_session(monkeypatch, repo=repo, bar_store=bar_store, instrument_id=instrument_id)
    _stub_backfiller(monkeypatch)

    result = await backfill_ops._features_backfill(
        _settings(tmp_path),
        _request(tmp_path, dry_run=True),
    )

    assert result.payload["dry_run"] is True
    assert result.payload["vectors_would_store"] == 1
    assert repo.vectors == []
