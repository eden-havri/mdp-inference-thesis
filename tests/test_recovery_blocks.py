from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdp_inference.artifacts import load_grid_spec
from mdp_inference.gridworld import gridworld_mdp
from mdp_inference.mdp import BranchedFiniteHorizonMDP, FiniteHorizonMDP
from mdp_inference.recovery_blocks import build_modal_path_recovery_blocks


def _dense_copy(compact) -> FiniteHorizonMDP:
    transition = np.zeros(
        (compact.num_states, compact.num_actions, compact.num_states),
        dtype=np.float64,
    )
    reward_mass = np.zeros_like(transition)
    for state in range(compact.num_states):
        for action in range(compact.num_actions):
            for branch in range(compact.num_branches):
                probability = compact.probability[state, action, branch]
                if probability <= 0.0:
                    continue
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


def _medium_path_guide(num_states: int, num_actions: int) -> np.ndarray:
    path = [
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
        233, 217, 201, 185, 169, 153, 137, 138, 122, 123,
        107, 91, 75, 59, 60, 44, 45, 29, 13, 14,
    ]
    successors = path[1:] + [15]
    guide = np.full((num_states, num_actions), 1.0 / num_actions, dtype=np.float64)
    for state, successor in zip(path, successors):
        delta = successor - state
        action = {1: 0, -16: 1}[delta]
        guide[state] = 0.0
        guide[state, action] = 1.0
    return guide


def _block_states(mdp, blocks: np.ndarray, block: int) -> list[int]:
    decisions = np.asarray(mdp.decision_states or (), dtype=np.int64)
    return decisions[blocks[block]].tolist()


def test_medium_recovery_blocks_match_expected_states_and_storage_formats() -> None:
    root = Path(__file__).resolve().parents[1]
    spec, _ = load_grid_spec(root / "experiments" / "maps" / "medium-seed0.json")
    compact = gridworld_mdp(spec)
    dense = _dense_copy(compact)
    guide = _medium_path_guide(compact.num_states, compact.num_actions)

    compact_blocks = build_modal_path_recovery_blocks(compact, guide)
    dense_blocks = build_modal_path_recovery_blocks(dense, guide)
    decision_guide = guide[np.asarray(compact.decision_states, dtype=np.int64)]

    assert compact_blocks.dtype == np.bool_
    assert compact_blocks.shape == (17, compact.num_decisions)
    assert np.array_equal(compact_blocks, dense_blocks)
    assert np.array_equal(
        compact_blocks,
        build_modal_path_recovery_blocks(compact, decision_guide),
    )
    assert _block_states(compact, compact_blocks, 0) == [
        13, 14, 29, 44, 45, 59, 60, 75, 91, 107,
        122, 123, 137, 138, 153, 169, 185, 201, 217, 233,
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
    ]
    assert [
        _block_states(compact, compact_blocks, block)
        for block in range(1, len(compact_blocks))
    ] == [
        [224, 225], [226, 227], [228, 230], [231, 232],
        [234, 250], [216, 218], [200, 202], [170, 186],
        [154, 168], [124, 136], [92, 108], [74, 76],
        [43, 58], [28, 61], [30, 46], [12],
    ]
    assert compact_blocks[1:].sum(axis=1).tolist() == [2] * 15 + [1]
    assert np.all(compact_blocks.sum(axis=0) <= 1)


def test_successor_probabilities_are_sorted_aggregated_and_copied() -> None:
    duplicated = BranchedFiniteHorizonMDP(
        next_state=np.asarray([[[1, 0, 1]], [[1, 1, 1]]]),
        probability=np.asarray([[[0.2, 0.5, 0.3]], [[1.0, 0.0, 0.0]]]),
        branch_reward=np.zeros((2, 1, 3), dtype=np.float64),
        initial=np.asarray([1.0, 0.0]),
        horizon=2,
        terminal=np.asarray([False, True]),
        decision_states=(0,),
    )
    duplicate_states, duplicate_probabilities = duplicated.successor_probabilities(0, 0)
    assert duplicate_states.tolist() == [0, 1]
    assert np.allclose(duplicate_probabilities, [0.5, 0.5])

    root = Path(__file__).resolve().parents[1]
    spec, _ = load_grid_spec(root / "experiments" / "maps" / "medium-seed0.json")
    compact = gridworld_mdp(spec)
    dense = _dense_copy(compact)

    for state in (13, 137, 240):
        for action in range(compact.num_actions):
            compact_states, compact_probabilities = compact.successor_probabilities(
                state, action
            )
            dense_states, dense_probabilities = dense.successor_probabilities(state, action)
            assert np.all(np.diff(compact_states) > 0)
            assert np.array_equal(compact_states, dense_states)
            assert np.allclose(compact_probabilities, dense_probabilities, atol=1e-15)

    states, probabilities = compact.successor_probabilities(240, 0)
    states[:] = 0
    probabilities[:] = 0.0
    repeated_states, repeated_probabilities = compact.successor_probabilities(240, 0)
    assert repeated_states.tolist() != states.tolist()
    assert np.isclose(repeated_probabilities.sum(), 1.0)


def test_recovery_block_builder_rejects_a_modal_cycle() -> None:
    transition = np.zeros((3, 2, 3), dtype=np.float64)
    transition[0, :, 1] = 1.0
    transition[1, :, 0] = 1.0
    transition[2, :, 2] = 1.0
    mdp = FiniteHorizonMDP(
        transition=transition,
        reward=np.zeros_like(transition),
        initial=np.asarray([1.0, 0.0, 0.0]),
        horizon=4,
        terminal=np.asarray([False, False, True]),
        decision_states=(0, 1),
    )

    with pytest.raises(ValueError, match="cycles"):
        build_modal_path_recovery_blocks(mdp, np.full((2, 2), 0.5))
