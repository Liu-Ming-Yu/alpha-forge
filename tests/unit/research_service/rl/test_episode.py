from __future__ import annotations

from dataclasses import dataclass

import pytest

from quant_platform.research.rl import (
    ActionInterpreter,
    EpisodeRunner,
    Policy,
    Reward,
    StateInterpreter,
)


class _CounterSimulator:
    def __init__(self, target: int) -> None:
        self.value = 0
        self.target = target

    def step(self, action: int) -> None:
        self.value += action

    def get_state(self) -> int:
        return self.value

    def done(self) -> bool:
        return self.value >= self.target


class _IdentityState(StateInterpreter[int, int]):
    def interpret(self, simulator_state: int) -> int:
        return simulator_state


class _PositiveAction(ActionInterpreter[int, int, int]):
    def validate_action(self, action: int) -> None:
        if action <= 0:
            raise ValueError("action must be positive")

    def interpret(self, simulator_state: int, action: int) -> int:
        del simulator_state
        return action


@dataclass(frozen=True)
class _IncrementPolicy(Policy[int, int]):
    amount: int = 1

    def act(self, observation: int) -> int:
        del observation
        return self.amount


class _StateReward(Reward[int]):
    def reward(self, simulator_state: int) -> float:
        return float(simulator_state)


def test_episode_runner_collects_finite_trajectory() -> None:
    runner = EpisodeRunner(
        simulator=_CounterSimulator(target=3),
        state_interpreter=_IdentityState(),
        action_interpreter=_PositiveAction(),
        policy=_IncrementPolicy(),
        reward=_StateReward(),
    )

    result = runner.run()

    assert [step.state for step in result.steps] == [1, 2, 3]
    assert result.total_reward == 6.0
    assert result.final_state == 3
    assert result.steps[-1].done is True


def test_episode_runner_honors_max_steps() -> None:
    runner = EpisodeRunner(
        simulator=_CounterSimulator(target=10),
        state_interpreter=_IdentityState(),
        action_interpreter=_PositiveAction(),
        policy=_IncrementPolicy(),
        max_steps=2,
    )

    result = runner.run()

    assert len(result.steps) == 2
    assert result.final_state == 2


def test_episode_runner_rejects_invalid_max_steps() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        EpisodeRunner(
            simulator=_CounterSimulator(target=1),
            state_interpreter=_IdentityState(),
            action_interpreter=_PositiveAction(),
            policy=_IncrementPolicy(),
            max_steps=0,
        )
