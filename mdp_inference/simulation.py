from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .mdp import FiniteHorizonMDP


@dataclass(frozen=True)
class Episode:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    @property
    def total_return(self) -> float:
        return float(self.rewards.sum())

    @property
    def length(self) -> int:
        return int(len(self.rewards))


def simulate_episode(
    mdp: FiniteHorizonMDP,
    action_probabilities: Callable[[int], np.ndarray],
    rng: np.random.Generator,
) -> Episode:
    state = mdp.sample_initial(float(rng.random()))
    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[int] = []
    dones: list[bool] = []
    for time in range(mdp.horizon):
        if mdp.terminal[state]:
            break
        probabilities = np.asarray(action_probabilities(state), dtype=np.float64)
        if probabilities.shape != (mdp.num_actions,):
            raise ValueError("action distribution has the wrong shape")
        if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("action probabilities must be nonnegative and sum to one")
        action = int(rng.choice(mdp.num_actions, p=probabilities))
        next_state, reward = mdp.sample_transition(state, action, float(rng.random()))
        done = bool(mdp.terminal[next_state] or time == mdp.horizon - 1)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        state = next_state
        if done:
            break
    return Episode(
        states=np.asarray(states, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float64),
        next_states=np.asarray(next_states, dtype=np.int64),
        dones=np.asarray(dones, dtype=bool),
    )


def discounted_returns_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64)
    result = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result
