from __future__ import annotations

import numpy as np

from .mdp import FiniteHorizonMDP


RECOVERY_BLOCK_CONSTRUCTION = "modal_path_paired_fringe_v2"
RECOVERY_FRINGE_BLOCK_SIZE = 2


def _decision_guide(
    mdp: FiniteHorizonMDP, guide_probabilities: np.ndarray
) -> np.ndarray:
    guide = np.asarray(guide_probabilities, dtype=np.float64)
    decision_states = np.asarray(mdp.decision_states or (), dtype=np.int64)
    if guide.shape == (mdp.num_states, mdp.num_actions):
        guide = guide[decision_states]
    elif guide.shape != (mdp.num_decisions, mdp.num_actions):
        raise ValueError("guide_probabilities must have shape [D, A] or [S, A]")
    if not np.all(np.isfinite(guide)) or np.any(guide < 0.0):
        raise ValueError("guide probabilities must be finite and nonnegative")
    if not np.allclose(guide.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("guide probability rows must sum to one")
    return guide


def build_modal_path_recovery_blocks(
    mdp: FiniteHorizonMDP,
    guide_probabilities: np.ndarray,
) -> np.ndarray:
    """Build deterministic path and one-hop recovery blocks.

    ``B0`` follows the action-wise modal guide from the modal initial state,
    taking the modal successor at each step until a terminal state is reached.
    The remaining blocks partition every representable nonterminal one-hop
    successor of a ``B0`` state under every action, excluding ``B0`` itself.
    Successors are ordered by their first occurrence while scanning the modal
    path from start to goal, then actions and successor IDs in ascending order.
    Consecutive pairs form small recovery blocks (the final block may be a
    singleton), limiting the reverse-density penalty in sparse-reward modes.

    The result is a boolean ``[B, D]`` catalog indexed by
    ``mdp.decision_states``. Ties are resolved by the smallest action or state
    ID, making the construction independent of RNG state and storage format.
    """

    if mdp.num_decisions == 0:
        raise ValueError("recovery blocks require at least one decision state")
    guide = _decision_guide(mdp, guide_probabilities)
    decision_states = np.asarray(mdp.decision_states or (), dtype=np.int64)
    state_to_decision = mdp.state_to_decision

    state = int(np.argmax(mdp.initial))
    if bool(mdp.terminal[state]):
        raise ValueError("modal initial state is terminal")

    path_states: list[int] = []
    seen: set[int] = set()
    while not bool(mdp.terminal[state]):
        if state in seen:
            raise ValueError("modal guide path cycles before reaching a terminal state")
        decision = state_to_decision.get(state)
        if decision is None:
            raise ValueError("modal guide path visits a nonterminal non-decision state")
        seen.add(state)
        path_states.append(state)

        action = int(np.argmax(guide[decision]))
        successors, probabilities = mdp.successor_probabilities(state, action)
        if successors.size == 0:
            raise ValueError("modal guide path has no positive-probability successor")
        # successor_probabilities returns sorted IDs, so np.argmax resolves a
        # probability tie in favor of the smallest successor ID.
        state = int(successors[int(np.argmax(probabilities))])

    path_set = set(path_states)
    neighbor_states: set[int] = set()
    neighbor_order: list[int] = []
    for path_state in path_states:
        for action in range(mdp.num_actions):
            successors, _ = mdp.successor_probabilities(path_state, action)
            for successor in successors:
                successor_state = int(successor)
                if (
                    successor_state not in path_set
                    and not bool(mdp.terminal[successor_state])
                    and successor_state in state_to_decision
                ):
                    if successor_state not in neighbor_states:
                        neighbor_states.add(successor_state)
                        neighbor_order.append(successor_state)

    if not neighbor_states:
        raise ValueError("modal guide path has an empty one-hop recovery block")

    num_fringe_blocks = int(
        np.ceil(len(neighbor_order) / RECOVERY_FRINGE_BLOCK_SIZE)
    )
    blocks = np.zeros((1 + num_fringe_blocks, mdp.num_decisions), dtype=bool)
    blocks[0, [state_to_decision[state] for state in path_states]] = True
    for block_index, start in enumerate(
        range(0, len(neighbor_order), RECOVERY_FRINGE_BLOCK_SIZE),
        start=1,
    ):
        chunk = neighbor_order[start : start + RECOVERY_FRINGE_BLOCK_SIZE]
        blocks[block_index, [state_to_decision[state] for state in chunk]] = True
    return blocks
