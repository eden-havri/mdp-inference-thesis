from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def _goal_mask(mdp: FiniteHorizonMDP, goal_states: Sequence[int]) -> np.ndarray:
    """Validate goal indices and return their Boolean state mask."""

    raw_goals = np.asarray(goal_states)
    if raw_goals.ndim != 1 or raw_goals.size == 0:
        raise ValueError("goal_states must be a nonempty one-dimensional sequence")
    if not np.issubdtype(raw_goals.dtype, np.integer):
        raise ValueError("goal_states must contain integer state indices")
    goals = raw_goals.astype(np.int64, copy=False)
    if np.any((goals < 0) | (goals >= mdp.num_states)):
        raise ValueError("goal state is outside the state space")
    mask = np.zeros(mdp.num_states, dtype=bool)
    mask[goals] = True
    return mask


def _transition_probability_backups(
    mdp: FiniteHorizonMDP,
    next_probability: np.ndarray,
) -> np.ndarray:
    """Return E[next_probability[S']] for every state-action pair.

    This deliberately excludes rewards and discounting.  It supports both the
    dense ``[S, A, S]`` representation and the compact fixed-branch
    ``[S, A, B]`` representation used by GridWorld.
    """

    next_probability = np.asarray(next_probability, dtype=np.float64)
    if next_probability.shape != (mdp.num_states,):
        raise ValueError("next_probability must have shape [S]")
    if hasattr(mdp, "next_state") and hasattr(mdp, "probability"):
        return np.sum(
            mdp.probability * next_probability[mdp.next_state],
            axis=2,
        )
    return np.sum(mdp.transition * next_probability[None, None, :], axis=2)


def evaluate_deterministic_policy_goal_probability(
    mdp: FiniteHorizonMDP,
    policy: Sequence[int],
    goal_states: Sequence[int],
) -> float:
    """Exact probability that a stationary deterministic policy hits a goal.

    A trajectory is successful when it starts in, or first enters, any state
    in ``goal_states`` at or before the finite horizon.  Rewards and ``gamma``
    do not affect this event probability.
    """

    policy_array = mdp.validate_policy(policy).copy()
    goals = _goal_mask(mdp, goal_states)
    policy_array[mdp.terminal | goals] = 0
    states = np.arange(mdp.num_states)
    probability = goals.astype(np.float64)
    for _ in range(mdp.horizon):
        action_probabilities = _transition_probability_backups(mdp, probability)
        probability = action_probabilities[states, policy_array]
        probability[goals] = 1.0
        probability[mdp.terminal & ~goals] = 0.0
    return float(np.dot(mdp.initial, probability))


def evaluate_stationary_stochastic_policy_goal_probability(
    mdp: FiniteHorizonMDP,
    probabilities: np.ndarray,
    goal_states: Sequence[int],
) -> float:
    """Exact goal-hitting probability under stationary stochastic execution.

    The action distribution is sampled anew on every visit.  This differs
    from first sampling one deterministic policy and committing to it for the
    whole episode; callers should report those two execution semantics
    separately.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (mdp.num_states, mdp.num_actions):
        raise ValueError("probabilities must have shape [S, A]")
    for state in mdp.decision_states or ():
        row = probabilities[state]
        if (
            not np.isfinite(row).all()
            or np.any(row < 0.0)
            or not np.isclose(row.sum(), 1.0)
        ):
            raise ValueError("decision-state rows must be finite distributions")

    goals = _goal_mask(mdp, goal_states)
    probabilities = probabilities.copy()
    probabilities[mdp.terminal | goals] = 0.0
    probabilities[mdp.terminal | goals, 0] = 1.0
    probability = goals.astype(np.float64)
    for _ in range(mdp.horizon):
        action_probabilities = _transition_probability_backups(mdp, probability)
        probability = np.sum(probabilities * action_probabilities, axis=1)
        probability[goals] = 1.0
        probability[mdp.terminal & ~goals] = 0.0
    return float(np.dot(mdp.initial, probability))


def evaluate_nonstationary_deterministic_policy_goal_probability(
    mdp: FiniteHorizonMDP,
    policy: np.ndarray,
    goal_states: Sequence[int],
) -> float:
    """Exact goal-hitting probability for a time-indexed deterministic policy."""

    policy = np.asarray(policy, dtype=np.int64)
    if policy.shape != (mdp.horizon, mdp.num_states):
        raise ValueError("policy must have shape [H, S]")
    decision_states = np.asarray(mdp.decision_states or (), dtype=np.int64)
    if decision_states.size:
        decisions = policy[:, decision_states]
        if np.any((decisions < 0) | (decisions >= mdp.num_actions)):
            raise ValueError("policy contains an invalid decision action")

    goals = _goal_mask(mdp, goal_states)
    policy = policy.copy()
    policy[:, mdp.terminal | goals] = 0
    states = np.arange(mdp.num_states)
    probability = goals.astype(np.float64)
    for time in range(mdp.horizon - 1, -1, -1):
        action_probabilities = _transition_probability_backups(mdp, probability)
        probability = action_probabilities[states, policy[time]]
        probability[goals] = 1.0
        probability[mdp.terminal & ~goals] = 0.0
    return float(np.dot(mdp.initial, probability))


def finite_horizon_optimal_goal_reaching_control(
    mdp: FiniteHorizonMDP,
    goal_states: Sequence[int],
) -> DynamicProgrammingSolution:
    """Exact nonstationary policy maximizing goal probability by the horizon."""

    goals = _goal_mask(mdp, goal_states)
    values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=np.float64)
    values[:, goals] = 1.0
    policy = np.zeros((mdp.horizon, mdp.num_states), dtype=np.int64)
    for time in range(mdp.horizon - 1, -1, -1):
        action_probabilities = _transition_probability_backups(mdp, values[time + 1])
        policy[time] = np.argmax(action_probabilities, axis=1)
        values[time] = np.max(action_probabilities, axis=1)
        values[time, goals] = 1.0
        values[time, mdp.terminal & ~goals] = 0.0
    return DynamicProgrammingSolution(values, policy, float(np.dot(mdp.initial, values[0])))


def uniform_random_policy_value(mdp: FiniteHorizonMDP) -> float:
    probabilities = np.full((mdp.num_states, mdp.num_actions), 1.0 / mdp.num_actions)
    return evaluate_stationary_stochastic_policy(mdp, probabilities)
