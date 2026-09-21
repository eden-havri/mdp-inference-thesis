from __future__ import annotations

import json

import numpy as np
import pytest

from mdp_inference.artifacts import write_grid_spec
from mdp_inference.experiment import ExperimentConfig, run_experiment
from mdp_inference.gridworld import GridWorldSpec, generate_gridworld, gridworld_mdp, state_of
from mdp_inference.mdp import FiniteHorizonMDP
from mdp_inference.oracles import (
    evaluate_deterministic_policy_goal_probability,
    evaluate_nonstationary_deterministic_policy_goal_probability,
    evaluate_stationary_stochastic_policy_goal_probability,
    finite_horizon_optimal_goal_reaching_control,
)


def _retry_mdp(horizon: int = 2) -> FiniteHorizonMDP:
    """Action 0 succeeds immediately; action 1 leaves the agent at start."""

    transition = np.zeros((2, 2, 2), dtype=np.float64)
    transition[0, 0, 1] = 1.0
    transition[0, 1, 0] = 1.0
    transition[1, :, 1] = 1.0
    return FiniteHorizonMDP(
        transition=transition,
        reward=np.zeros_like(transition),
        initial=np.asarray([1.0, 0.0]),
        horizon=horizon,
        terminal=np.asarray([False, True]),
    )


def _dense_copy(compact) -> FiniteHorizonMDP:
    transition = np.zeros(
        (compact.num_states, compact.num_actions, compact.num_states), dtype=np.float64
    )
    reward_mass = np.zeros_like(transition)
    for state in range(compact.num_states):
        for action in range(compact.num_actions):
            for branch in range(compact.num_branches):
                probability = compact.probability[state, action, branch]
                successor = compact.next_state[state, action, branch]
                transition[state, action, successor] += probability
                reward_mass[state, action, successor] += (
                    probability * compact.branch_reward[state, action, branch]
                )
    reward = np.divide(
        reward_mass,
        transition,
        out=np.zeros_like(reward_mass),
        where=transition > 0.0,
    )
    return FiniteHorizonMDP(
        transition=transition,
        reward=reward,
        initial=compact.initial,
        horizon=compact.horizon,
        terminal=compact.terminal,
        decision_states=compact.decision_states,
        gamma=compact.gamma,
    )


def test_stationary_deterministic_and_stochastic_goal_probabilities() -> None:
    mdp = _retry_mdp(horizon=2)
    always_try = np.asarray([0, 99])  # Terminal-state action is intentionally ignored.
    never_try = np.asarray([1, 0])
    probabilities = np.asarray([[0.25, 0.75], [np.nan, np.nan]])

    assert evaluate_deterministic_policy_goal_probability(mdp, always_try, (1,)) == 1.0
    assert evaluate_deterministic_policy_goal_probability(mdp, never_try, (1,)) == 0.0
    assert np.isclose(
        evaluate_stationary_stochastic_policy_goal_probability(mdp, probabilities, (1,)),
        1.0 - 0.75**2,
    )


def test_nonstationary_evaluation_and_success_oracle() -> None:
    mdp = _retry_mdp(horizon=2)
    policy = np.asarray([[1, 17], [0, -3]])
    assert evaluate_nonstationary_deterministic_policy_goal_probability(mdp, policy, (1,)) == 1.0
    solution = finite_horizon_optimal_goal_reaching_control(mdp, (1,))
    assert solution.initial_value == 1.0
    assert evaluate_nonstationary_deterministic_policy_goal_probability(
        mdp, solution.policy, (1,)
    ) == solution.initial_value


def test_goal_probability_matches_dense_and_compact_gridworlds() -> None:
    spec = generate_gridworld("smoke", seed=9)
    compact = gridworld_mdp(spec)
    dense = _dense_copy(compact)
    goals = tuple(state_of(spec, cell) for cell in spec.goals)
    rng = np.random.default_rng(41)
    probabilities = rng.dirichlet(np.ones(compact.num_actions), size=compact.num_states)
    policy = np.argmax(probabilities, axis=1).astype(np.int64)

    assert np.isclose(
        evaluate_stationary_stochastic_policy_goal_probability(compact, probabilities, goals),
        evaluate_stationary_stochastic_policy_goal_probability(dense, probabilities, goals),
        atol=1e-12,
    )
    assert np.isclose(
        evaluate_deterministic_policy_goal_probability(compact, policy, goals),
        evaluate_deterministic_policy_goal_probability(dense, policy, goals),
        atol=1e-12,
    )
    assert np.isclose(
        finite_horizon_optimal_goal_reaching_control(compact, goals).initial_value,
        finite_horizon_optimal_goal_reaching_control(dense, goals).initial_value,
        atol=1e-12,
    )


def test_goal_probability_validation() -> None:
    mdp = _retry_mdp()
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_deterministic_policy_goal_probability(mdp, np.zeros(2, dtype=int), ())
    with pytest.raises(ValueError, match="outside"):
        evaluate_deterministic_policy_goal_probability(mdp, np.zeros(2, dtype=int), (2,))
    with pytest.raises(ValueError, match="finite distributions"):
        evaluate_stationary_stochastic_policy_goal_probability(
            mdp, np.asarray([[np.nan, np.nan], [0.5, 0.5]]), (1,)
        )


def test_random_experiment_records_goal_probability_metrics(tmp_path) -> None:
    spec = GridWorldSpec(
        rows=2,
        cols=2,
        start=(1, 0),
        goals=((0, 1),),
        slip_probability=0.0,
        horizon=3,
    )
    map_path = tmp_path / "map.json"
    write_grid_spec(map_path, spec)
    output = run_experiment(
        ExperimentConfig(
            run_id="goal-metric-smoke",
            method="random",
            map_path=str(map_path),
            output_root=str(tmp_path / "results"),
            evaluation_policy_samples=8,
            transition_budget=1,
        )
    )
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    for key in (
        "goal_reaching_probability",
        "marginal_action_goal_probability",
        "committed_policy_goal_probability",
        "map_policy_goal_probability",
        "random_marginal_goal_probability",
        "random_committed_goal_probability",
        "oracle_goal_probability",
    ):
        assert 0.0 <= result[key] <= 1.0
    assert result["goal_reaching_probability"] == result[
        "committed_policy_goal_probability"
    ]
    assert result["goal_reaching_probability_semantics"] == "policy_sample_then_commit"
    assert result["value"] == result["committed_policy_value"]
    assert result["primary_value_semantics"] == "policy_sample_then_commit"
