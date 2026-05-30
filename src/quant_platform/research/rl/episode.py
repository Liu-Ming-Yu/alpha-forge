"""Episode runner that composes simulator, interpreters, policy, and reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from quant_platform.research.rl.contracts import (
    ObservationT,
    Policy,
    PolicyActionT,
    Simulator,
    SimulatorActionT,
    StateT,
)

if TYPE_CHECKING:
    from quant_platform.research.rl.interpreters import ActionInterpreter, StateInterpreter
    from quant_platform.research.rl.rewards import Reward


class AuxiliaryInfoCollector(Generic[StateT]):
    """Optional collector for state-derived diagnostics."""

    def __call__(self, simulator_state: StateT) -> dict[str, object]:
        return self.collect(simulator_state)

    def collect(self, simulator_state: StateT) -> dict[str, object]:
        """Return auxiliary information for the current state."""
        raise NotImplementedError


@dataclass(frozen=True)
class EpisodeStep(Generic[StateT, ObservationT, PolicyActionT, SimulatorActionT]):
    """One transition collected by :class:`EpisodeRunner`."""

    state: StateT
    observation: ObservationT
    policy_action: PolicyActionT
    simulator_action: SimulatorActionT
    reward: float
    done: bool
    aux_info: dict[str, object]


@dataclass(frozen=True)
class EpisodeResult(Generic[StateT, ObservationT, PolicyActionT, SimulatorActionT]):
    """Full trajectory summary."""

    steps: tuple[EpisodeStep[StateT, ObservationT, PolicyActionT, SimulatorActionT], ...]
    total_reward: float
    final_state: StateT


class EpisodeRunner(Generic[StateT, ObservationT, PolicyActionT, SimulatorActionT]):
    """Run one finite policy trajectory against a simulator."""

    def __init__(
        self,
        *,
        simulator: Simulator[object, StateT, SimulatorActionT],
        state_interpreter: StateInterpreter[StateT, ObservationT],
        action_interpreter: ActionInterpreter[StateT, PolicyActionT, SimulatorActionT],
        policy: Policy[ObservationT, PolicyActionT],
        reward: Reward[StateT] | None = None,
        aux_info_collector: AuxiliaryInfoCollector[StateT] | None = None,
        max_steps: int | None = None,
    ) -> None:
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        self._simulator = simulator
        self._state_interpreter = state_interpreter
        self._action_interpreter = action_interpreter
        self._policy = policy
        self._reward = reward
        self._aux_info_collector = aux_info_collector
        self._max_steps = max_steps

    def run(self) -> EpisodeResult[StateT, ObservationT, PolicyActionT, SimulatorActionT]:
        """Run until the simulator is done or ``max_steps`` is reached."""
        steps: list[EpisodeStep[StateT, ObservationT, PolicyActionT, SimulatorActionT]] = []
        total_reward = 0.0
        while not self._simulator.done():
            if self._max_steps is not None and len(steps) >= self._max_steps:
                break
            state = self._simulator.get_state()
            observation = self._state_interpreter(state)
            policy_action = self._policy.act(observation)
            simulator_action = self._action_interpreter(state, policy_action)
            self._simulator.step(simulator_action)
            next_state = self._simulator.get_state()
            step_reward = self._reward(next_state) if self._reward is not None else 0.0
            total_reward += step_reward
            aux_info = (
                self._aux_info_collector(next_state) if self._aux_info_collector is not None else {}
            )
            steps.append(
                EpisodeStep(
                    state=next_state,
                    observation=observation,
                    policy_action=policy_action,
                    simulator_action=simulator_action,
                    reward=step_reward,
                    done=self._simulator.done(),
                    aux_info=aux_info,
                )
            )
        return EpisodeResult(
            steps=tuple(steps),
            total_reward=total_reward,
            final_state=self._simulator.get_state(),
        )


__all__ = [
    "AuxiliaryInfoCollector",
    "EpisodeResult",
    "EpisodeRunner",
    "EpisodeStep",
]
