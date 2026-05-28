"""Session composition package interface."""

from quant_platform.application.runtime.state import CycleResult, Session, SessionDrawdownGuard
from quant_platform.bootstrap.session.factory import build_session
from quant_platform.bootstrap.session.public_factory import (
    create_ib_paper_session_impl,
    create_live_session_impl,
    create_paper_session_impl,
    maybe_attach_v2_orchestrator,
)
from quant_platform.engines.session.runtime import (
    hydrate_session_state,
    model_registry_preflight,
)
from quant_platform.engines.session.strategy_cycle import (
    durable_kill_switch,
    record_nav_snapshot,
    run_strategy_cycle,
    run_strategy_cycle_unlocked,
    strategy_cycle_lock,
)

__all__ = [
    "CycleResult",
    "Session",
    "SessionDrawdownGuard",
    "build_session",
    "create_ib_paper_session_impl",
    "create_live_session_impl",
    "create_paper_session_impl",
    "durable_kill_switch",
    "hydrate_session_state",
    "maybe_attach_v2_orchestrator",
    "model_registry_preflight",
    "record_nav_snapshot",
    "run_strategy_cycle",
    "run_strategy_cycle_unlocked",
    "strategy_cycle_lock",
]
