"""Runtime session state containers.

These live in the application layer rather than ``core``: they reference
concrete service-controller ports and runtime adapters. They carry no wiring,
so both ``bootstrap`` (which composes a ``Session``) and ``engines`` (which
runs one) depend on this module without depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from quant_platform.application.runtime.handles import (
    SESSION_ATTRS,
    BrokerHandles,
    ControllerHandles,
    PolicyHandles,
    RepositoryHandles,
    controller_handles_from_kwargs,
    repository_handles_from_kwargs,
)
from quant_platform.core.domain.production import ExecutionTacticPolicy

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        Clock,
    )
    from quant_platform.core.domain.orders import FillEvent
    from quant_platform.core.domain.orders.intent import OrderIntent
    from quant_platform.core.domain.portfolio import PortfolioTarget
    from quant_platform.core.domain.signals import SignalScore


@dataclass
class CycleResult:
    """Summary of one strategy rebalance cycle."""

    signals: list[SignalScore]
    target: PortfolioTarget | None
    approved: list[OrderIntent]
    rejected: list[OrderIntent]
    submitted_ids: list[uuid.UUID]
    fills: list[FillEvent]
    proposal: object = None


class SessionDrawdownGuard:
    """Tracks intra-session NAV high-water-mark and fires on drawdown breach."""

    def __init__(self, max_drawdown: Decimal) -> None:
        self._hwm: Decimal | None = None
        self._limit = max_drawdown

    def update_and_check(self, nav: Decimal) -> tuple[bool, Decimal]:
        """Update HWM and return ``(ok, drawdown_fraction)``."""
        if nav > (self._hwm or Decimal("0")):
            self._hwm = nav
        if self._limit >= Decimal("0") or self._hwm is None or self._hwm == Decimal("0"):
            return True, Decimal("0")
        if self._hwm < Decimal("0"):
            raise RuntimeError(
                f"Negative high-water-mark ({self._hwm}): drawdown fraction is undefined. "
                "Activate kill switch and investigate account state before resuming."
            )
        drawdown = (self._hwm - nav) / self._hwm
        return drawdown <= -self._limit, drawdown


_SessionDrawdownGuard = SessionDrawdownGuard


@dataclass(init=False)
class Session:
    """Runtime session grouped by adapters, policies, controllers, and stores."""

    settings: PlatformSettings
    clock: Clock
    brokers: BrokerHandles
    policies: PolicyHandles
    controllers: ControllerHandles
    repositories: RepositoryHandles
    _state_hydrated: bool = field(default=False, repr=False)

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "settings", kwargs.pop("settings"))
        object.__setattr__(self, "clock", kwargs.pop("clock"))
        object.__setattr__(
            self,
            "brokers",
            BrokerHandles(
                broker=kwargs.pop("broker"),
                account_broker=kwargs.pop("account_broker"),
                trading_broker=kwargs.pop("trading_broker"),
                lifecycle_feed=kwargs.pop("lifecycle_feed"),
            ),
        )
        object.__setattr__(
            self,
            "policies",
            PolicyHandles(
                cash_engine=kwargs.pop("cash_engine"),
                risk_policy=kwargs.pop("risk_policy"),
                execution_policy=kwargs.pop("execution_policy"),
                pretrade_gate=kwargs.pop("pretrade_gate"),
                risk_limits=kwargs.pop("risk_limits"),
                execution_tactic_policy=kwargs.pop(
                    "execution_tactic_policy",
                    ExecutionTacticPolicy(),
                ),
                kill_switch_store=kwargs.pop("kill_switch_store", None),
                drawdown_guard=kwargs.pop("drawdown_guard", None),
            ),
        )
        object.__setattr__(self, "controllers", controller_handles_from_kwargs(kwargs))
        object.__setattr__(self, "repositories", repository_handles_from_kwargs(kwargs))
        object.__setattr__(self, "_state_hydrated", kwargs.pop("_state_hydrated", False))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown Session fields: {unknown}")

    def __getattr__(self, name: str) -> Any:
        location = SESSION_ATTRS.get(name)
        if location is None:
            raise AttributeError(name)
        group_name, attr_name = location
        return getattr(getattr(self, group_name), attr_name)

    def __setattr__(self, name: str, value: Any) -> None:
        location = SESSION_ATTRS.get(name)
        if location is None or not hasattr(self, location[0]):
            object.__setattr__(self, name, value)
            return
        group_name, attr_name = location
        setattr(getattr(self, group_name), attr_name, value)
