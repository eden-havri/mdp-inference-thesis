from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteHorizonMDP


@dataclass(frozen=True)
class DynamicProgrammingSolution:
    value: np.ndarray
    policy: np.ndarray
    initial_value: float


def finite_horizon_optimal_control(mdp: FiniteHorizonMDP) -> DynamicProgrammingSolution:
    """Exact nonstationary finite-horizon expected-return optimum."""

    values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=np.float64)
    policy = np.zeros((mdp.horizon, mdp.num_states), dtype=np.int64)
    for time in range(mdp.horizon - 1, -1, -1):
        action_values = mdp.action_backups(values[time + 1])
        policy[time] = np.argmax(action_values, axis=1)
        values[time] = np.max(action_values, axis=1)
        values[time, mdp.terminal] = 0.0
    return DynamicProgrammingSolution(values, policy, float(np.dot(mdp.initial, values[0])))


def evaluate_stationary_stochastic_policy(mdp: FiniteHorizonMDP, probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (mdp.num_states, mdp.num_actions):
        raise ValueError("probabilities must have shape [S, A]")
    for state in mdp.decision_states or ():
        if np.any(probabilities[state] < 0.0) or not np.isclose(probabilities[state].sum(), 1.0):
            raise ValueError("decision-state rows must be distributions")

    value = np.zeros(mdp.num_states, dtype=np.float64)
    for _ in range(mdp.horizon):
        action_values = mdp.action_backups(value)
        value = np.sum(probabilities * action_values, axis=1)
        value[mdp.terminal] = 0.0
    return float(np.dot(mdp.initial, value))


def uniform_random_policy_value(mdp: FiniteHorizonMDP) -> float:
    probabilities = np.full((mdp.num_states, mdp.num_actions), 1.0 / mdp.num_actions)
    return evaluate_stationary_stochastic_policy(mdp, probabilities)
