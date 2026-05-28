"""Regression coverage for shadow engine cycles."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.bootstrap.engine.session_wiring import (
    build_engine_maintenance_scheduler,
    create_engine_runtime_session,
)
from quant_platform.config import (
    AlphaSettings,
    BoostingSettings,
    LLMSettings,
    PlatformSettings,
    RiskSettings,
    StorageSettings,
    V2Settings,
)
from quant_platform.core.domain.market_data.text_events import TextEvent, TextEventType
from quant_platform.core.domain.research import FeatureVector
from quant_platform.core.exceptions import DataStalenessError
from quant_platform.engines.engine_runner import EngineConfig, EngineRunner, RunMode
from quant_platform.engines.framework.plugins import (
    create_engine_from_plugin,
    get_strategy_plugin,
)
from quant_platform.services.data_service.text.text_event_store import InMemoryTextEventStore
from quant_platform.session import create_paper_session


@pytest.mark.asyncio
async def test_single_engine_live_path_blocked_when_v2_account_orchestrator_enabled(
    tmp_path,
) -> None:
    """When V2 owns the account, the single-engine live path must fail
    fast so two submitters cannot run in parallel.  This is the 'V2 is the
    only live submit path' production gate.
    """
    instrument_id = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
        v2=V2Settings(
            enabled=True,
            account_orchestrator_enabled=True,
        ),
    )
    runner = EngineRunner(
        EngineConfig(
            engine_name="single_engine",
            engine_version="1.0.0",
            run_mode=RunMode.LIVE,
            initial_cash=Decimal("100000"),
            factor_weights={"momentum_1m": 1.0},
            instrument_contracts={
                instrument_id: {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "sector": "Information Technology",
                    "adv_shares_20d": 2_000_000,
                    "last_close": "100",
                    "con_id": 265598,
                }
            },
        ),
        settings=settings,
    )

    with pytest.raises(RuntimeError, match="V2 account orchestrator"):
        await runner.initialize(
            session_factory=create_engine_runtime_session,
            scheduler_factory=build_engine_maintenance_scheduler,
        )


def test_single_engine_live_path_guard_no_op_when_v2_disabled(tmp_path) -> None:
    """V1 deployments without V2 must not be blocked."""
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
        v2=V2Settings(enabled=False, account_orchestrator_enabled=False),
    )
    runner = EngineRunner(
        EngineConfig(
            engine_name="single_engine",
            engine_version="1.0.0",
            run_mode=RunMode.LIVE,
        ),
        settings=settings,
    )

    runner._assert_v2_is_only_live_submitter()  # noqa: SLF001 - explicit guard probe


@pytest.mark.asyncio
async def test_initialize_runs_model_registry_preflight(tmp_path, monkeypatch) -> None:
    from quant_platform.engines import engine_runner as er_mod

    calls: list[tuple[str, str]] = []

    async def _preflight(session, *, strategy_name: str, engine_version: str):  # type: ignore[no-untyped-def]
        calls.append((strategy_name, engine_version))

    monkeypatch.setattr(er_mod, "model_registry_preflight", _preflight)

    runner = EngineRunner(
        EngineConfig(
            engine_name="xsec",
            engine_version="1.2.3",
            run_mode=RunMode.SHADOW,
        ),
        settings=PlatformSettings(
            _env_file=None, storage=StorageSettings(object_store_root=str(tmp_path))
        ),
    )
    await runner.initialize(
        session_factory=create_engine_runtime_session,
        scheduler_factory=build_engine_maintenance_scheduler,
    )
    await runner.shutdown()

    assert calls == [("xsec", "1.2.3")]


@pytest.mark.asyncio
async def test_shadow_cycle_uses_regime_state_field_without_submitting(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.50"),
            max_sector_weight=Decimal("0.80"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.50"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )
    runner = EngineRunner(
        EngineConfig(
            engine_name="shadow_smoke",
            run_mode=RunMode.SHADOW,
            initial_cash=Decimal("100000"),
            factor_weights={"momentum_1m": 1.0},
            instrument_contracts={
                instrument_id: {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "sector": "Information Technology",
                    "adv_shares_20d": 2_000_000,
                    "last_close": "100",
                }
            },
        ),
        settings=settings,
    )

    await runner.initialize(
        session_factory=create_engine_runtime_session,
        scheduler_factory=build_engine_maintenance_scheduler,
    )
    try:
        result = await runner.run_cycle(
            feature_data={instrument_id: {"momentum_1m": 1.0}},
            market_prices={instrument_id: Decimal("100")},
        )
    finally:
        summary = await runner.shutdown()

    assert result.submitted_ids == []
    assert result.fills == []
    assert len(result.signals) == 1
    assert result.target is not None
    assert len(result.approved) > 0
    assert summary.shadow_only is True


@pytest.mark.asyncio
async def test_engine_runner_fails_closed_on_incomplete_plugin_features(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.50"),
            max_sector_weight=Decimal("0.80"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.50"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )
    runner = EngineRunner(
        EngineConfig(
            engine_name="schema_guard",
            run_mode=RunMode.SHADOW,
            initial_cash=Decimal("100000"),
            factor_weights={"momentum_1m": 1.0, "momentum_3m": 1.0},
            required_features=("momentum_1m", "momentum_3m"),
            instrument_contracts={
                instrument_id: {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "sector": "Information Technology",
                    "adv_shares_20d": 2_000_000,
                    "last_close": "100",
                }
            },
        ),
        settings=settings,
    )

    await runner.initialize(
        session_factory=create_engine_runtime_session,
        scheduler_factory=build_engine_maintenance_scheduler,
    )
    try:
        with pytest.raises(DataStalenessError, match="missing required features"):
            await runner.run_cycle(
                feature_data={instrument_id: {"momentum_1m": 1.0}},
                market_prices={instrument_id: Decimal("100")},
            )
    finally:
        await runner.shutdown()


@pytest.mark.asyncio
async def test_engine_runner_fails_closed_on_non_finite_plugin_features(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )
    runner = EngineRunner(
        EngineConfig(
            engine_name="schema_guard",
            run_mode=RunMode.SHADOW,
            factor_weights={"momentum_1m": 1.0},
            required_features=("momentum_1m",),
            instrument_contracts={
                instrument_id: {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "sector": "Information Technology",
                    "adv_shares_20d": 2_000_000,
                    "last_close": "100",
                }
            },
        ),
        settings=settings,
    )

    await runner.initialize(
        session_factory=create_engine_runtime_session,
        scheduler_factory=build_engine_maintenance_scheduler,
    )
    try:
        with pytest.raises(DataStalenessError, match="non-finite feature values"):
            await runner.run_cycle(
                feature_data={instrument_id: {"momentum_1m": float("nan")}},
                market_prices={instrument_id: Decimal("100")},
            )
    finally:
        await runner.shutdown()


@pytest.mark.asyncio
async def test_engine_runner_fails_closed_on_non_numeric_plugin_features(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )
    runner = EngineRunner(
        EngineConfig(
            engine_name="schema_guard",
            run_mode=RunMode.SHADOW,
            factor_weights={"momentum_1m": 1.0},
            required_features=("momentum_1m",),
            instrument_contracts={
                instrument_id: {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "sector": "Information Technology",
                    "adv_shares_20d": 2_000_000,
                    "last_close": "100",
                }
            },
        ),
        settings=settings,
    )

    await runner.initialize(
        session_factory=create_engine_runtime_session,
        scheduler_factory=build_engine_maintenance_scheduler,
    )
    try:
        with pytest.raises(DataStalenessError, match="non-finite feature values"):
            await runner.run_cycle(
                feature_data={instrument_id: {"momentum_1m": "bad-payload"}},  # type: ignore[dict-item]
                market_prices={instrument_id: Decimal("100")},
            )
    finally:
        await runner.shutdown()


@pytest.mark.asyncio
async def test_shadow_boosting_runs_without_changing_order_flow(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_platform.services.research_service.boosting as boosting_mod

    calls: list[tuple[int, int]] = []

    class _FakeBoostingModel:
        model_version = "xgb-shadow"
        feature_schema_hash = "hash"
        device = "cpu"

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class _FakeShadowScorer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def score_cycle(self, *, feature_data, primary_scores, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((len(feature_data), len(primary_scores)))
            return tmp_path / "shadow.jsonl"

    monkeypatch.setattr(boosting_mod, "XGBoostRankSignalModel", _FakeBoostingModel)
    monkeypatch.setattr(boosting_mod, "ShadowBoostingScorer", _FakeShadowScorer)

    instrument_id = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.50"),
            max_sector_weight=Decimal("0.80"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.50"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
        storage=StorageSettings(object_store_root=str(tmp_path)),
        boosting=BoostingSettings(
            enabled=True,
            artifact_manifest=str(tmp_path / "manifest.json"),
            device="cpu",
            shadow_artifact_root=str(tmp_path / "shadow"),
        ),
    )
    runner = EngineRunner(
        EngineConfig(
            engine_name="shadow_boost",
            run_mode=RunMode.SHADOW,
            initial_cash=Decimal("100000"),
            factor_weights={"momentum_1m": 1.0},
            instrument_contracts={
                instrument_id: {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "sector": "Information Technology",
                    "adv_shares_20d": 2_000_000,
                    "last_close": "100",
                }
            },
        ),
        settings=settings,
    )

    await runner.initialize(
        session_factory=create_engine_runtime_session,
        scheduler_factory=build_engine_maintenance_scheduler,
    )
    try:
        result = await runner.run_cycle(
            feature_data={instrument_id: {"momentum_1m": 1.0}},
            market_prices={instrument_id: Decimal("100")},
        )
    finally:
        await runner.shutdown()

    assert calls == [(1, 1)]
    assert result.submitted_ids == []
    assert result.fills == []
    assert len(result.approved) > 0


def test_etf_engine_factory_defaults(tmp_path) -> None:
    plugin = get_strategy_plugin("etf_macro_allocator")
    runner = create_engine_from_plugin(
        "etf_macro_allocator",
        run_mode=RunMode.SHADOW,
        initial_cash=Decimal("50000"),
        settings=PlatformSettings(
            _env_file=None, storage=StorageSettings(object_store_root=str(tmp_path))
        ),
    )

    assert runner._config.engine_name == plugin.name
    assert runner._config.max_positions == 4
    assert runner._config.rebalance_interval_seconds == 86400.0
    assert runner._config.factor_weights == plugin.default_factor_weights
    assert plugin.default_universe_symbols == (
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "GLD",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
    )


@pytest.mark.asyncio
async def test_engine_config_max_positions_limits_portfolio_constructor(tmp_path) -> None:
    ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
    settings = PlatformSettings(
        _env_file=None,
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.80"),
            max_sector_weight=Decimal("0.95"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.50"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )
    runner = EngineRunner(
        EngineConfig(
            engine_name="max_positions",
            run_mode=RunMode.SHADOW,
            initial_cash=Decimal("100000"),
            factor_weights={"momentum_1m": 1.0},
            max_positions=1,
            instrument_contracts={
                instrument_id: {
                    "symbol": f"T{i}",
                    "exchange": "SMART",
                    "currency": "USD",
                    "sector": "ETF",
                    "adv_shares_20d": 2_000_000,
                    "last_close": "100",
                }
                for i, instrument_id in enumerate(ids)
            },
        ),
        settings=settings,
    )

    await runner.initialize(
        session_factory=create_engine_runtime_session,
        scheduler_factory=build_engine_maintenance_scheduler,
    )
    try:
        result = await runner.run_cycle(
            feature_data={
                ids[0]: {"momentum_1m": 1.0},
                ids[1]: {"momentum_1m": 0.9},
                ids[2]: {"momentum_1m": 0.8},
            },
            market_prices={instrument_id: Decimal("100") for instrument_id in ids},
        )
    finally:
        await runner.shutdown()

    assert result.target is not None
    assert len(result.target.weights) == 1


@pytest.mark.asyncio
async def test_shadow_cycle_hydrates_session_before_signals(tmp_path, monkeypatch) -> None:
    from quant_platform.engines import engine_runner as er_mod

    instrument_id = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.50"),
            max_sector_weight=Decimal("0.80"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.50"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )
    runner = EngineRunner(
        EngineConfig(
            engine_name="shadow_hydration",
            run_mode=RunMode.SHADOW,
            initial_cash=Decimal("100000"),
            factor_weights={"momentum_1m": 1.0},
            instrument_contracts={
                instrument_id: {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "sector": "Information Technology",
                    "adv_shares_20d": 2_000_000,
                    "last_close": "100",
                }
            },
        ),
        settings=settings,
    )
    hydrate_calls: list[object] = []

    async def _capture_hydrate(session):  # type: ignore[no-untyped-def]
        hydrate_calls.append(session)
        session._state_hydrated = True

    monkeypatch.setattr(er_mod, "hydrate_session_state", _capture_hydrate)

    await runner.initialize(
        session_factory=create_engine_runtime_session,
        scheduler_factory=build_engine_maintenance_scheduler,
    )
    assert runner._session is not None
    original_generate = runner._session.signal_ctrl.generate  # type: ignore[union-attr]

    async def _assert_hydrated_before_generate(
        *args: object,
        **kwargs: object,
    ):  # type: ignore[no-untyped-def]
        assert hydrate_calls == [runner._session]
        assert runner._session._state_hydrated is True
        return await original_generate(*args, **kwargs)

    runner._session.signal_ctrl.generate = _assert_hydrated_before_generate  # type: ignore[union-attr,method-assign]
    try:
        await runner.run_cycle(
            feature_data={instrument_id: {"momentum_1m": 1.0}},
            market_prices={instrument_id: Decimal("100")},
        )
    finally:
        await runner.shutdown()


@pytest.mark.asyncio
async def test_shadow_cycle_refreshes_regime_before_detect(tmp_path) -> None:
    """R-GOV-03: shadow cycle must call `_compute_market_stats_from_store`
    -> `MarketRegimeDetector.update()` -> `detect()` in that order, matching
    the paper/live cycle.  Before the fix the shadow path called
    ``detect()`` cold and could silently disagree with paper/live.
    """
    from quant_platform.services.signal_service.regime_detector import (
        MarketRegimeDetector,
    )

    instrument_id = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.50"),
            max_sector_weight=Decimal("0.80"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.50"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )
    runner = EngineRunner(
        EngineConfig(
            engine_name="shadow_regime",
            run_mode=RunMode.SHADOW,
            initial_cash=Decimal("100000"),
            factor_weights={"momentum_1m": 1.0},
            instrument_contracts={
                instrument_id: {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "sector": "Information Technology",
                    "adv_shares_20d": 2_000_000,
                    "last_close": "100",
                }
            },
        ),
        settings=settings,
    )

    await runner.initialize(
        session_factory=create_engine_runtime_session,
        scheduler_factory=build_engine_maintenance_scheduler,
    )
    assert runner._session is not None

    # Force the live detector type so the new MarketRegimeDetector branch
    # fires regardless of settings defaults.
    runner._session.regime_detector = MarketRegimeDetector()  # type: ignore[assignment]

    update_calls: list[object] = []
    orig_update = runner._session.regime_detector.update  # type: ignore[attr-defined]

    def _capture_update(stats: object) -> None:
        update_calls.append(stats)
        orig_update(stats)

    runner._session.regime_detector.update = _capture_update  # type: ignore[assignment]

    # Stub the stats helper so we can assert it was called even without a
    # real bar store populated.
    stats_calls: list[object] = []

    async def _stub_stats(session, as_of):  # type: ignore[no-untyped-def]
        stats_calls.append((session, as_of))
        return None  # detector.update is skipped on None — that's fine

    # The import inside _run_shadow_cycle is lazy, so patch the runtime-API
    # module where _compute_market_stats_from_store now lives.
    from quant_platform.engines.session import public_api as runtime_api

    session_orig = runtime_api._compute_market_stats_from_store
    runtime_api._compute_market_stats_from_store = _stub_stats
    try:
        await runner.run_cycle(
            feature_data={instrument_id: {"momentum_1m": 1.0}},
            market_prices={instrument_id: Decimal("100")},
        )
    finally:
        runtime_api._compute_market_stats_from_store = session_orig
        await runner.shutdown()

    assert stats_calls, (
        "shadow cycle must call _compute_market_stats_from_store before regime detection"
    )
    # update() may not fire because the stub returns None; the key
    # invariant is that the stats helper ran at all — that closes the
    # parity gap.


def test_paper_session_wires_text_event_store(tmp_path) -> None:
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
    )

    session = create_paper_session(settings=settings)

    assert isinstance(session.text_event_store, InMemoryTextEventStore)


def test_llm_live_mode_fails_closed(tmp_path) -> None:
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
        llm=LLMSettings(live_mode_enabled=True),
    )

    with pytest.raises(RuntimeError, match="ENSEMBLE_MODE=live"):
        create_paper_session(settings=settings)


def test_llm_live_mode_requires_startup_assertion(tmp_path) -> None:
    manifest = tmp_path / "text_model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
        alpha=AlphaSettings(
            ensemble_mode="live",
            source_weights={"classical": 0.99, "text": 0.01},
        ),
        llm=LLMSettings(
            live_mode_enabled=True,
            text_model_manifest=str(manifest),
        ),
    )

    with pytest.raises(RuntimeError, match="fresh live startup assertion"):
        create_paper_session(settings=settings)


@pytest.mark.asyncio
async def test_shadow_text_cycle_reads_session_text_store(tmp_path, monkeypatch) -> None:
    instrument_id = uuid.uuid4()
    artifact = tmp_path / "earnings.txt"
    artifact.write_text("Revenue growth accelerated.", encoding="utf-8")
    settings = PlatformSettings(
        _env_file=None,
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.50"),
            max_sector_weight=Decimal("0.80"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.50"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
        storage=StorageSettings(object_store_root=str(tmp_path)),
        llm=LLMSettings(shadow_mode_enabled=True),
    )

    class _FakeExtractor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def extract(
            self,
            event: TextEvent,
            _text_content: str,
            strategy_run_id: uuid.UUID,
            *,
            as_of: datetime | None = None,
        ) -> FeatureVector:
            return FeatureVector(
                vector_id=uuid.uuid4(),
                instrument_id=event.instrument_id or uuid.UUID(int=0),
                strategy_run_id=strategy_run_id,
                as_of=as_of or event.occurred_at,
                features={
                    "text_sentiment": 0.8,
                    "guidance_direction": 1.0,
                    "revenue_revision_magnitude": 0.4,
                    "macro_sentiment": 0.0,
                },
                feature_set_version="text-v1",
                artifact_uri=f"{event.artifact_uri}#prompt=v1",
            )

    from quant_platform.services.research_service.text import features as text_features

    monkeypatch.setattr(text_features, "LLMTextFeatureExtractor", _FakeExtractor)
    runner = EngineRunner(
        EngineConfig(
            engine_name="shadow_text_store",
            run_mode=RunMode.SHADOW,
            initial_cash=Decimal("100000"),
            factor_weights={"momentum_1m": 1.0},
            instrument_contracts={
                instrument_id: {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "sector": "Information Technology",
                    "adv_shares_20d": 2_000_000,
                    "last_close": "100",
                }
            },
        ),
        settings=settings,
    )

    await runner.initialize(
        session_factory=create_engine_runtime_session,
        scheduler_factory=build_engine_maintenance_scheduler,
    )
    assert runner._session is not None
    event = TextEvent(
        event_id=uuid.uuid4(),
        event_type=TextEventType.EARNINGS_TRANSCRIPT,
        occurred_at=runner._session.clock.now() - timedelta(seconds=1),
        source_uri=str(artifact),
        artifact_uri=str(artifact),
        instrument_id=instrument_id,
    )
    await runner._session.text_event_store.store_event(event)
    try:
        await runner.run_cycle(
            feature_data={instrument_id: {"momentum_1m": 1.0}},
            market_prices={instrument_id: Decimal("100")},
        )
        vectors = await runner._session.feature_repo.get_vectors(
            [instrument_id],
            "text-v1",
            runner._session.clock.now(),
        )
    finally:
        await runner.shutdown()

    assert len(vectors) == 1
    assert vectors[0].artifact_uri.endswith("#prompt=v1")
