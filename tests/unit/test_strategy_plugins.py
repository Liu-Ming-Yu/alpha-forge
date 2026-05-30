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
    assert list_strategy_plugins() == (
        "arm_g",
        "arm_q",
        "cross_sectional_equity",
        "etf_macro_allocator",
    )

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


def test_arm_q_is_arm_g_plus_conviction_weighting() -> None:
    # Q = G's exact alpha (same pv_formulaic family, same frozen weights, same
    # selection) but conviction-sized — the production lead. Only the constructor
    # differs from arm_g.
    q = get_strategy_plugin("arm_q")
    g = get_strategy_plugin("arm_g")
    assert q.name == "arm_q_pv_formulaic_conviction_v1"
    assert q.feature_set_version == "pv-formulaic-live-v1"
    assert dict(q.default_factor_weights) == dict(g.default_factor_weights)  # identical alpha
    assert q.conviction_shrinkage == 0.25

    settings = PlatformSettings(_env_file=None)
    q_ctor = q.build_portfolio_constructor(settings, 30)
    g_ctor = g.build_portfolio_constructor(settings, 30)
    # Q builds a conviction-weighted constructor; G stays equal-weight.
    assert isinstance(q_ctor, LongOnlyPortfolioConstructor)
    assert q_ctor._conviction_shrinkage == 0.25
    assert g_ctor._conviction_shrinkage is None


def test_arm_q_runner_binds_pv_formulaic_version() -> None:
    runner = create_engine_from_plugin(
        "arm_q",
        run_mode=RunMode.SHADOW,
        initial_cash=Decimal("50000"),
        settings=PlatformSettings(_env_file=None),
    )
    assert runner._config.plugin_name == "arm_q_pv_formulaic_conviction_v1"
    assert runner._config.feature_set_version == "pv-formulaic-live-v1"
    assert runner._config.max_positions == 30


def test_arm_q_declares_promoted_governance_identity() -> None:
    # Naming-gap fix: Q preflights against the PROMOTED research-arm record so
    # ib-paper matches without QP__RISK__REQUIRE_REGISTERED_MODEL_MATCH=false.
    q = get_strategy_plugin("arm_q")
    assert q.registered_model_name == "long_only_top30_pv_formulaic_streakdial_conviction"
    assert q.registered_model_version == "ic-weighted-non-negative"
    # Every other plugin keeps the default (None) → preflight uses the engine identity.
    assert get_strategy_plugin("arm_g").registered_model_name is None
    assert get_strategy_plugin("arm_g").registered_model_version is None
    assert get_strategy_plugin("cross_sectional_equity").registered_model_name is None


def test_arm_q_runner_threads_governance_identity_into_config() -> None:
    runner = create_engine_from_plugin(
        "arm_q",
        run_mode=RunMode.SHADOW,
        initial_cash=Decimal("50000"),
        settings=PlatformSettings(_env_file=None),
    )
    assert (
        runner._config.registered_model_name == "long_only_top30_pv_formulaic_streakdial_conviction"
    )
    assert runner._config.registered_model_version == "ic-weighted-non-negative"
    # A plugin without an override leaves them None (engine-identity preflight).
    g = create_engine_from_plugin(
        "arm_g",
        run_mode=RunMode.SHADOW,
        initial_cash=Decimal("50000"),
        settings=PlatformSettings(_env_file=None),
    )
    assert g._config.registered_model_name is None
    assert g._config.registered_model_version is None


def test_execution_rebalance_threshold_default_and_wiring() -> None:
    # Finding #1: the order-planner rebalance threshold is configurable (was a
    # hardcoded 1%, which skipped a 30-name conviction book's low-weight tail).
    from quant_platform.bootstrap.session.public_api import create_paper_session

    base = PlatformSettings(_env_file=None)
    assert base.execution.rebalance_threshold == Decimal("0.01")  # back-compat default

    settings = base.model_copy(
        update={
            "execution": base.execution.model_copy(
                update={"rebalance_threshold": Decimal("0.002")}
            )
        }
    )
    session = create_paper_session(settings=settings)
    assert session.order_planner is not None
    assert session.order_planner._rebalance_threshold == Decimal("0.002")


def test_unknown_plugin_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown strategy plugin"):
        get_strategy_plugin("direct_broker_algo")
