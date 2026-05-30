"""Lightweight RL building blocks for research workflows.

This package adapts qlib's simulator/interpreter/reward shape without pulling
gym or tianshou into the platform. It is intended for research-time policy
experiments such as formulaic-alpha search, not live broker routing.

Two consumption patterns share these contracts:

* **Externally-driven search** — the current consumer. ``PolicySearch``
  (``features/formulaic/mining/policy_search.py``) reuses :class:`Simulator`,
  :class:`Policy`, :class:`StateInterpreter`, and :class:`ActionInterpreter`,
  but runs its own loop because candidate *fitness* is supplied by the mining
  driver (``mine_alphas``), not computed from simulator state. It deliberately
  does **not** use :class:`EpisodeRunner` / :class:`Reward`.
* **Self-contained episodes** — :class:`EpisodeRunner` plus the
  :class:`Reward` / :class:`RewardCombination` stack drive a full trajectory
  when the reward is a function of simulator state alone. This is the reusable
  substrate for future learned-policy experiments; it has no consumer in the
  mining path today and is covered by its own unit tests.
"""

from __future__ import annotations

from quant_platform.research.rl.contracts import Policy, Simulator
from quant_platform.research.rl.episode import (
    AuxiliaryInfoCollector,
    EpisodeResult,
    EpisodeRunner,
    EpisodeStep,
)
from quant_platform.research.rl.interpreters import ActionInterpreter, StateInterpreter
from quant_platform.research.rl.rewards import Reward, RewardCombination, RewardComponent

__all__ = [
    "ActionInterpreter",
    "AuxiliaryInfoCollector",
    "EpisodeResult",
    "EpisodeRunner",
    "EpisodeStep",
    "Policy",
    "Reward",
    "RewardCombination",
    "RewardComponent",
    "Simulator",
    "StateInterpreter",
]
