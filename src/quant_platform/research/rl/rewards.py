"""Reward contracts for qlib-style research simulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from quant_platform.research.rl.contracts import StateT


class Reward(Generic[StateT]):
    """Callable reward component over a simulator state."""

    def __call__(self, simulator_state: StateT) -> float:
        return float(self.reward(simulator_state))

    def reward(self, simulator_state: StateT) -> float:
        """Compute this component's reward."""
        raise NotImplementedError


@dataclass(frozen=True)
class RewardComponent:
    """One named contribution inside a combined reward."""

    name: str
    raw_value: float
    weight: float
    weighted_value: float


class RewardCombination(Reward[StateT]):
    """Weighted sum of multiple reward components."""

    def __init__(self, rewards: dict[str, tuple[Reward[StateT], float]]) -> None:
        if not rewards:
            raise ValueError("RewardCombination requires at least one reward")
        self._rewards = dict(rewards)
        self._last_components: tuple[RewardComponent, ...] = ()

    @property
    def last_components(self) -> tuple[RewardComponent, ...]:
        """Reward breakdown from the most recent call."""
        return self._last_components

    def reward(self, simulator_state: StateT) -> float:
        components: list[RewardComponent] = []
        total = 0.0
        for name, (reward, weight) in self._rewards.items():
            raw = reward(simulator_state)
            weighted = raw * float(weight)
            total += weighted
            components.append(
                RewardComponent(
                    name=name,
                    raw_value=raw,
                    weight=float(weight),
                    weighted_value=weighted,
                )
            )
        self._last_components = tuple(components)
        return total


__all__ = ["Reward", "RewardCombination", "RewardComponent"]
