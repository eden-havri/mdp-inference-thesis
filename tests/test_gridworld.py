from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mdp_inference.artifacts import load_grid_spec
from mdp_inference.gridworld import (
    GRID_DIFFICULTIES,
    GridWorldSpec,
    characterize_gridworld,
    generate_gridworld,
    gridworld_mdp,
    has_path,
)
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


def test_path_checks_respect_terminal_hazards() -> None:
    blocked = GridWorldSpec(
        rows=2,
        cols=3,
        start=(0, 0),
        goals=((0, 2),),
        walls=((1, 0), (1, 1), (1, 2)),
        hazards=((0, 1),),
        slip_probability=0.0,
        horizon=4,
    )
    assert not has_path(blocked)


def test_grid_characteristics_count_routes_and_bottlenecks() -> None:
    open_grid = GridWorldSpec(
        rows=2,
        cols=2,
        start=(1, 0),
        goals=((0, 1),),
        slip_probability=0.0,
        horizon=4,
    )
    characteristics = characterize_gridworld(open_grid)
    assert characteristics.shortest_path_length == 2
    assert characteristics.shortest_path_count == 2
    assert characteristics.reachable_fraction == 1.0
    assert characteristics.single_cell_bottlenecks == ()

    corridor = GridWorldSpec(
        rows=2,
        cols=3,
        start=(0, 0),
        goals=((0, 2),),
        walls=((1, 0), (1, 1), (1, 2)),
        slip_probability=0.0,
        horizon=4,
    )
    corridor_characteristics = characterize_gridworld(corridor)
    assert corridor_characteristics.shortest_path_count == 1
    assert corridor_characteristics.single_cell_bottlenecks == ((0, 1),)


def test_frozen_map_characteristics_are_audited() -> None:
    root = Path(__file__).resolve().parents[1]
    records = json.loads(
        (root / "experiments" / "maps" / "characteristics.json").read_text(
            encoding="utf-8"
        )
    )
    for record in records:
        spec, _ = load_grid_spec(root / "experiments" / "maps" / record["map"])
        observed = characterize_gridworld(spec)
        assert record["size"] == f"{spec.rows}x{spec.cols}"
        assert np.isclose(
            record["wall_fraction"], len(spec.walls) / float(spec.rows * spec.cols)
        )
        assert np.isclose(record["reachable_fraction"], observed.reachable_fraction)
        assert record["shortest_path_length"] == observed.shortest_path_length
        assert record["shortest_path_count"] == observed.shortest_path_count
        assert record["single_cell_bottleneck_count"] == len(
            observed.single_cell_bottlenecks
        )

    manifest = json.loads(
        (root / "experiments" / "maps" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    identities = {(record["tier"], record["seed"]) for record in manifest}
    assert len(identities) == len(manifest)
    for record in manifest:
        relative = Path(record["path"].replace("\\", "/"))
        _, digest = load_grid_spec(root / relative)
        assert digest == record["sha256"]


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
