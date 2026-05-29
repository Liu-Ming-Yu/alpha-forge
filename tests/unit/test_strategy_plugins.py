from __future__ import annotations

from decimal import Decimal

import pytest

from quant_platform.config import PlatformSettings
from quant_platform.engines.engine_runner import ExecutionBackend, RunMode
from quant_platform.engines.framework.plugins import (
    create_engine_from_plugin,
    get_strategy_plugin,
    list_strategy_plugins,
)
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel


def test_builtin_plugins_expose_required_contracts() -> None:
    assert list_strategy_plugins() == ("arm_g", "cross_sectional_equity", "etf_macro_allocator")

    plugin = get_strategy_plugin("cross_sectional_equity")
    model = plugin.build_signal_model({"momentum_1m": 1.0}, "test")
    constructor = plugin.build_portfolio_constructor(PlatformSettings(_env_file=None), 3)

    assert plugin.name == "cross_sectional_equity_v1"
    assert plugin.feature_spec.required_features
    assert isinstance(model, LinearWeightSignalModel)
    assert isinstance(constructor, LongOnlyPortfolioConstructor)


def test_plugin_runner_carries_plugin_metadata() -> None:
    runner = create_engine_from_plugin(
        "etf_macro_allocator",
        run_mode=RunMode.SHADOW,
        initial_cash=Decimal("1000"),
        settings=PlatformSettings(_env_file=None),
    )

    assert runner._config.plugin_name == "etf_macro_allocator_v1"
    assert runner._config.feature_set_name == "classical-etf-macro"
    assert runner._config.execution_backend == ExecutionBackend.SIMULATED
    assert "trend_quality_63d" in runner._config.required_features
    # Backward compat: a close-family plugin leaves feature_set_version empty so the
    # engine falls back to the default (close) family version.
    assert runner._config.feature_set_version == ""


def test_arm_g_plugin_binds_pv_formulaic_family() -> None:
    plugin = get_strategy_plugin("arm_g")
    assert plugin.name == "arm_g_pv_formulaic_v1"
    assert plugin.feature_spec.version == "pv-formulaic-live-v1"
    assert plugin.feature_set_version == "pv-formulaic-live-v1"
    # G's 20 frozen IC weights are the live pv_formulaic feature names; sum to 1.0.
    assert len(plugin.default_factor_weights) == 20
    assert sum(plugin.default_factor_weights.values()) == pytest.approx(1.0)
    assert set(plugin.feature_spec.required_features) == set(plugin.default_factor_weights)
    assert "ret_252d" in plugin.feature_spec.required_features
    assert "wq_alpha_041" in plugin.feature_spec.required_features


def test_arm_g_runner_computes_pv_formulaic_version() -> None:
    runner = create_engine_from_plugin(
        "arm_g",
        run_mode=RunMode.SHADOW,
        initial_cash=Decimal("50000"),
        settings=PlatformSettings(_env_file=None),
    )

    assert runner._config.plugin_name == "arm_g_pv_formulaic_v1"
    assert runner._config.feature_set_name == "pv-formulaic-live"
    # The version the engine schedules its feature job under → resolves to the
    # pv_formulaic family (not the default close family).
    assert runner._config.feature_set_version == "pv-formulaic-live-v1"
    assert runner._config.max_positions == 30
    assert "mom_12_1" in runner._config.required_features


def test_unknown_plugin_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown strategy plugin"):
        get_strategy_plugin("direct_broker_algo")
