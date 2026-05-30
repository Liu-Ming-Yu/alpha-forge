"""Core RL contracts inspired by qlib's simulator-policy boundary."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

InitialStateT = TypeVar("InitialStateT")
StateT = TypeVar("StateT")
SimulatorActionT = TypeVar("SimulatorActionT")
ObservationT = TypeVar("ObservationT")
PolicyActionT = TypeVar("PolicyActionT")
_InitialStateCovariantT = TypeVar("_InitialStateCovariantT", covariant=True)
_StateCovariantT = TypeVar("_StateCovariantT", covariant=True)
_SimulatorActionContravariantT = TypeVar(
    "_SimulatorActionContravariantT",
    contravariant=True,
)
_ObservationContravariantT = TypeVar("_ObservationContravariantT", contravariant=True)
_PolicyActionCovariantT = TypeVar("_PolicyActionCovariantT", covariant=True)


@runtime_checkable
class Simulator(
    Protocol[_InitialStateCovariantT, _StateCovariantT, _SimulatorActionContravariantT]
):
    """Ephemeral simulator whose state changes only through ``step``.

    ``InitialStateT`` is part of the type contract for simulator factories even
    though it is not referenced in the protocol methods. This mirrors qlib's
    "seed creates a trajectory" lifecycle while keeping implementations free of
    inheritance.
    """

    def step(self, action: _SimulatorActionContravariantT) -> None:
        """Apply one simulator action."""
        ...

    def get_state(self) -> _StateCovariantT:
        """Return the current simulator state."""
        ...

    def done(self) -> bool:
        """Return whether this trajectory is complete."""
        ...


@runtime_checkable
class Policy(Protocol[_ObservationContravariantT, _PolicyActionCovariantT]):
    """Policy that maps observations to raw policy actions."""

    def act(self, observation: _ObservationContravariantT) -> _PolicyActionCovariantT:
        """Choose the next policy action."""
        ...


__all__ = [
    "InitialStateT",
    "ObservationT",
    "Policy",
    "PolicyActionT",
    "Simulator",
    "SimulatorActionT",
    "StateT",
]
