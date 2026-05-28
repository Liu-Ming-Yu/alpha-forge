"""Grouped runtime-session handles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quant_platform.core.domain.production import ExecutionTacticPolicy


@dataclass
class BrokerHandles:
    """Runtime broker adapters."""

    broker: Any
    account_broker: Any
    trading_broker: Any
    lifecycle_feed: Any | None


@dataclass
class PolicyHandles:
    """Runtime policy and guard adapters."""

    cash_engine: Any
    risk_policy: Any
    execution_policy: Any
    pretrade_gate: Any
    risk_limits: Any
    execution_tactic_policy: ExecutionTacticPolicy = field(default_factory=ExecutionTacticPolicy)
    kill_switch_store: Any | None = None
    drawdown_guard: Any | None = None


@dataclass
class ControllerHandles:
    """Runtime controllers and orchestrators."""

    approve_ctrl: Any
    submit_ctrl: Any
    recon_engine: Any
    recon_ctrl: Any
    coordinator: Any
    supervisor: Any | None = None
    signal_ctrl: Any | None = None
    portfolio_constructor: Any | None = None
    portfolio_ctrl: Any | None = None
    order_planner: Any | None = None
    regime_detector: Any | None = None
    account_orchestrator: Any | None = None


@dataclass
class RepositoryHandles:
    """Runtime repositories, stores, and catalogs."""

    event_bus: Any
    audit_sink: Any
    order_repo: Any
    position_repo: Any
    performance_repo: Any
    feature_repo: Any
    signal_contribution_repo: Any | None
    text_event_store: Any
    bar_store: Any
    contract_master: Any
    universe_manager: Any
    v2_order_state: Any | None = None
    v2_risk_repo: Any | None = None


SESSION_ATTRS = {
    "broker": ("brokers", "broker"),
    "account_broker": ("brokers", "account_broker"),
    "trading_broker": ("brokers", "trading_broker"),
    "lifecycle_feed": ("brokers", "lifecycle_feed"),
    "cash_engine": ("policies", "cash_engine"),
    "risk_policy": ("policies", "risk_policy"),
    "execution_policy": ("policies", "execution_policy"),
    "pretrade_gate": ("policies", "pretrade_gate"),
    "risk_limits": ("policies", "risk_limits"),
    "execution_tactic_policy": ("policies", "execution_tactic_policy"),
    "kill_switch_store": ("policies", "kill_switch_store"),
    "drawdown_guard": ("policies", "drawdown_guard"),
    "approve_ctrl": ("controllers", "approve_ctrl"),
    "submit_ctrl": ("controllers", "submit_ctrl"),
    "recon_engine": ("controllers", "recon_engine"),
    "recon_ctrl": ("controllers", "recon_ctrl"),
    "coordinator": ("controllers", "coordinator"),
    "supervisor": ("controllers", "supervisor"),
    "signal_ctrl": ("controllers", "signal_ctrl"),
    "portfolio_constructor": ("controllers", "portfolio_constructor"),
    "portfolio_ctrl": ("controllers", "portfolio_ctrl"),
    "order_planner": ("controllers", "order_planner"),
    "regime_detector": ("controllers", "regime_detector"),
    "account_orchestrator": ("controllers", "account_orchestrator"),
    "event_bus": ("repositories", "event_bus"),
    "audit_sink": ("repositories", "audit_sink"),
    "order_repo": ("repositories", "order_repo"),
    "position_repo": ("repositories", "position_repo"),
    "performance_repo": ("repositories", "performance_repo"),
    "feature_repo": ("repositories", "feature_repo"),
    "signal_contribution_repo": ("repositories", "signal_contribution_repo"),
    "text_event_store": ("repositories", "text_event_store"),
    "bar_store": ("repositories", "bar_store"),
    "contract_master": ("repositories", "contract_master"),
    "universe_manager": ("repositories", "universe_manager"),
    "v2_order_state": ("repositories", "v2_order_state"),
    "v2_risk_repo": ("repositories", "v2_risk_repo"),
}


def controller_handles_from_kwargs(kwargs: dict[str, Any]) -> ControllerHandles:
    """Build controller handles from Session keyword arguments."""
    return ControllerHandles(
        approve_ctrl=kwargs.pop("approve_ctrl"),
        submit_ctrl=kwargs.pop("submit_ctrl"),
        recon_engine=kwargs.pop("recon_engine"),
        recon_ctrl=kwargs.pop("recon_ctrl"),
        coordinator=kwargs.pop("coordinator"),
        supervisor=kwargs.pop("supervisor", None),
        signal_ctrl=kwargs.pop("signal_ctrl", None),
        portfolio_constructor=kwargs.pop("portfolio_constructor", None),
        portfolio_ctrl=kwargs.pop("portfolio_ctrl", None),
        order_planner=kwargs.pop("order_planner", None),
        regime_detector=kwargs.pop("regime_detector", None),
        account_orchestrator=kwargs.pop("account_orchestrator", None),
    )


def repository_handles_from_kwargs(kwargs: dict[str, Any]) -> RepositoryHandles:
    """Build repository handles from Session keyword arguments."""
    return RepositoryHandles(
        event_bus=kwargs.pop("event_bus"),
        audit_sink=kwargs.pop("audit_sink"),
        order_repo=kwargs.pop("order_repo"),
        position_repo=kwargs.pop("position_repo"),
        performance_repo=kwargs.pop("performance_repo"),
        feature_repo=kwargs.pop("feature_repo"),
        signal_contribution_repo=kwargs.pop("signal_contribution_repo"),
        text_event_store=kwargs.pop("text_event_store"),
        bar_store=kwargs.pop("bar_store"),
        contract_master=kwargs.pop("contract_master"),
        universe_manager=kwargs.pop("universe_manager"),
        v2_order_state=kwargs.pop("v2_order_state", None),
        v2_risk_repo=kwargs.pop("v2_risk_repo", None),
    )


__all__ = [
    "BrokerHandles",
    "ControllerHandles",
    "PolicyHandles",
    "RepositoryHandles",
    "SESSION_ATTRS",
    "controller_handles_from_kwargs",
    "repository_handles_from_kwargs",
]
