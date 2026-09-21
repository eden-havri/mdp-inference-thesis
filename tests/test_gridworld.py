from __future__ import annotations

import numpy as np

from mdp_inference.gridworld import GRID_DIFFICULTIES, generate_gridworld, gridworld_mdp, has_path
from mdp_inference.mdp import FiniteHorizonMDP
from mdp_inference.oracles import finite_horizon_optimal_control, uniform_random_policy_value


def test_generated_grids_are_solvable_and_reproducible() -> None:
    first = generate_gridworld("smoke", seed=42)
    second = generate_gridworld("smoke", seed=42)
    assert first == second
    assert has_path(first)
    assert len(first.walls) == round(
        GRID_DIFFICULTIES["smoke"].obstacle_density * first.rows * first.cols
    )


def test_gridworld_transition_rows_and_oracles() -> None:
    spec = generate_gridworld("smoke", seed=2)
    mdp = gridworld_mdp(spec)
    assert mdp.transition_model_kind == "fixed_branch"
    assert np.allclose(mdp.probability.sum(axis=2), 1.0)
    random_value = uniform_random_policy_value(mdp)
    optimum = finite_horizon_optimal_control(mdp)
    assert np.isfinite(random_value)
    assert np.isfinite(optimum.initial_value)
    assert optimum.initial_value >= random_value - 1e-10


def test_fixed_branch_grid_matches_dense_reference() -> None:
    spec = generate_gridworld("smoke", seed=7)
    compact = gridworld_mdp(spec)
    transition = np.zeros(
        (compact.num_states, compact.num_actions, compact.num_states), dtype=np.float64
    )
    reward = np.zeros_like(transition)
    for state in range(compact.num_states):
        for action in range(compact.num_actions):
            for branch in range(compact.num_branches):
                probability = compact.probability[state, action, branch]
                if probability == 0.0:
                    continue
                successor = compact.next_state[state, action, branch]
                transition[state, action, successor] += probability
                reward[state, action, successor] = compact.branch_reward[state, action, branch]
    dense = FiniteHorizonMDP(
        transition=transition,
        reward=reward,
        initial=compact.initial,
        horizon=compact.horizon,
        terminal=compact.terminal,
        decision_states=compact.decision_states,
        gamma=compact.gamma,
    )

    rng = np.random.default_rng(13)
    for _ in range(20):
        policy = np.zeros(compact.num_states, dtype=np.int64)
        policy[list(compact.decision_states or ())] = rng.integers(
            compact.num_actions, size=compact.num_decisions
        )
        assert np.isclose(compact.expected_return(policy), dense.expected_return(policy), atol=1e-12)
    for state in range(compact.num_states):
        for action in range(compact.num_actions):
            for u in (0.0, 0.1, 0.5, np.nextafter(1.0, 0.0)):
                assert compact.sample_transition(state, action, u) == dense.sample_transition(
                    state, action, u
                )
    assert np.isclose(
        finite_horizon_optimal_control(compact).initial_value,
        finite_horizon_optimal_control(dense).initial_value,
        atol=1e-12,
    )
    assert np.isclose(
        uniform_random_policy_value(compact),
        uniform_random_policy_value(dense),
        atol=1e-12,
    )
