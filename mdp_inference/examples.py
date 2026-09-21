from __future__ import annotations

import numpy as np

from .mdp import FiniteHorizonMDP


def zero_mean_variance_mdp(reward_magnitude: float = 2.0) -> FiniteHorizonMDP:
    """One-decision MDP exposing expected-return vs exponential utility.

    Action 0 terminates with reward zero. Action 1 terminates with rewards
    ``+reward_magnitude`` and ``-reward_magnitude`` with equal probability.
    Both actions have expected return zero.
    """

    transition = np.zeros((4, 2, 4), dtype=np.float64)
    reward = np.zeros_like(transition)

    transition[0, 0, 1] = 1.0
    transition[0, 1, 2] = 0.5
    transition[0, 1, 3] = 0.5
    reward[0, 1, 2] = reward_magnitude
    reward[0, 1, 3] = -reward_magnitude

    # Terminal rows are normalized here and made explicitly absorbing by the
    # FiniteHorizonMDP constructor as an additional invariant.
    for state in (1, 2, 3):
        transition[state, :, state] = 1.0

    return FiniteHorizonMDP(
        transition=transition,
        reward=reward,
        initial=np.array([1.0, 0.0, 0.0, 0.0]),
        horizon=1,
        terminal=np.array([False, True, True, True]),
        decision_states=(0,),
    )


def two_step_choice_mdp() -> FiniteHorizonMDP:
    """Small deterministic MDP with a unique best stationary policy."""

    # State 0 chooses whether to visit state 1 or terminate. At state 1,
    # action 1 produces the largest terminal reward.
    transition = np.zeros((3, 2, 3), dtype=np.float64)
    reward = np.zeros_like(transition)
    transition[0, 0, 1] = 1.0
    transition[0, 1, 2] = 1.0
    reward[0, 1, 2] = 0.5
    transition[1, 0, 2] = 1.0
    transition[1, 1, 2] = 1.0
    reward[1, 0, 2] = -0.25
    reward[1, 1, 2] = 2.0
    transition[2, :, 2] = 1.0
    return FiniteHorizonMDP(
        transition=transition,
        reward=reward,
        initial=np.array([1.0, 0.0, 0.0]),
        horizon=2,
        terminal=np.array([False, False, True]),
        decision_states=(0, 1),
    )
