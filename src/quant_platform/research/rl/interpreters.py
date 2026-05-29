"""State and action interpreters for RL research loops."""

from __future__ import annotations

from typing import Generic

from quant_platform.research.rl.contracts import (
    ObservationT,
    PolicyActionT,
    SimulatorActionT,
    StateT,
)


class StateInterpreter(Generic[StateT, ObservationT]):
    """Convert simulator state into a policy observation.

    Subclasses override :meth:`interpret` and optionally
    :meth:`validate_observation`. The call wrapper keeps validation in one place,
    matching qlib's interpreter pattern without depending on gym spaces.
    """

    def __call__(self, simulator_state: StateT) -> ObservationT:
        observation = self.interpret(simulator_state)
        self.validate_observation(observation)
        return observation

    def interpret(self, simulator_state: StateT) -> ObservationT:
        """Return the policy-facing observation for ``simulator_state``."""
        raise NotImplementedError

    def validate_observation(self, observation: ObservationT) -> None:
        """Validate an observation; default accepts every value."""


class ActionInterpreter(Generic[StateT, PolicyActionT, SimulatorActionT]):
    """Convert a raw policy action into a simulator action."""

    def __call__(self, simulator_state: StateT, action: PolicyActionT) -> SimulatorActionT:
        self.validate_action(action)
        return self.interpret(simulator_state, action)

    def interpret(self, simulator_state: StateT, action: PolicyActionT) -> SimulatorActionT:
        """Return the simulator-facing action for ``action``."""
        raise NotImplementedError

    def validate_action(self, action: PolicyActionT) -> None:
        """Validate a policy action; default accepts every value."""


__all__ = ["ActionInterpreter", "StateInterpreter"]
