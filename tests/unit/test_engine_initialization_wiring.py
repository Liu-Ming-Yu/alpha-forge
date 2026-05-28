"""Unit tests for engine initialization wiring helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.config import PlatformSettings, VolSizingSettings
from quant_platform.core.domain.research import RunType
from quant_platform.engines.framework.initialization_wiring import (
    build_engine_portfolio_constructor,
    build_signal_model,
    build_strategy_run,
    run_type_for_mode,
)
from quant_platform.engines.framework.types import EngineConfig, RunMode
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)
from quant_platform.services.portfolio_service.vol_sizing import (
    VolTargetedPortfolioConstructor,
)

_NOW = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return _NOW


def test_run_type_for_mode_maps_shadow_and_paper_to_paper() -> None:
    assert run_type_for_mode(RunMode.SHADOW) == RunType.PAPER
    assert run_type_for_mode(RunMode.PAPER) == RunType.PAPER
    assert run_type_for_mode(RunMode.LIVE) == RunType.LIVE


def test_build_strategy_run_captures_engine_config_snapshot() -> None:
    config = EngineConfig(
        engine_name="equity",
        engine_version="1.2.3",
        run_mode=RunMode.SHADOW,
        factor_weights={"momentum": 1.0},
        max_positions=7,
    )

    run = build_strategy_run(config=config, clock=_Clock())

    assert run.strategy_name == "equity"
    assert run.strategy_version == "1.2.3"
    assert run.run_type == RunType.PAPER
    assert run.config_snapshot == {
        "factor_weights": {"momentum": 1.0},
        "max_positions": 7,
        "run_mode": "shadow",
        "execution_backend": "simulated",
    }
    assert run.created_at == _NOW


def test_build_signal_model_prefers_configured_factory() -> None:
    sentinel = object()

    def _factory(weights: dict[str, float], version: str) -> object:
        assert weights == {"momentum": 1.0}
        assert version == "1.2.3"
        return sentinel

    model = build_signal_model(
        EngineConfig(
            engine_name="equity",
            engine_version="1.2.3",
            factor_weights={"momentum": 1.0},
            signal_model_factory=_factory,
        )
    )

    assert model is sentinel


def test_build_engine_portfolio_constructor_uses_default_long_only() -> None:
    constructor = build_engine_portfolio_constructor(
        config=EngineConfig(engine_name="equity", max_positions=3),
        settings=PlatformSettings(_env_file=None),
    )

    assert isinstance(constructor, LongOnlyPortfolioConstructor)


def test_build_engine_portfolio_constructor_uses_vol_target_when_enabled() -> None:
    constructor = build_engine_portfolio_constructor(
        config=EngineConfig(engine_name="equity", max_positions=3),
        settings=PlatformSettings(
            _env_file=None,
            vol_sizing=VolSizingSettings(
                enabled=True,
                vol_target_annualized=0.2,
                min_vol_floor=0.1,
            ),
        ),
    )

    assert isinstance(constructor, VolTargetedPortfolioConstructor)


def test_build_engine_portfolio_constructor_rejects_invalid_max_positions() -> None:
    with pytest.raises(ValueError, match="max_positions"):
        build_engine_portfolio_constructor(
            config=EngineConfig(
                engine_name="equity",
                max_positions=0,
                initial_cash=Decimal("100"),
            ),
            settings=PlatformSettings(_env_file=None),
        )
