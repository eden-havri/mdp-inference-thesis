from __future__ import annotations

import math

import numpy as np

from mdp_inference.exact import (
    exact_exponential_utility_posterior,
    exact_gibbs_posterior,
    exact_saa_posterior,
)
from mdp_inference.examples import two_step_choice_mdp, zero_mean_variance_mdp
from mdp_inference.mdp import FiniteHorizonMDP, TapeBank


def test_expected_return_target_does_not_prefer_variance() -> None:
    mdp = zero_mean_variance_mdp(2.0)
    posterior = exact_gibbs_posterior(mdp, beta=1.0)
    assert np.allclose(posterior.returns, [0.0, 0.0])
    assert np.allclose(posterior.probabilities, [0.5, 0.5])


def test_exponentiating_individual_returns_changes_the_target() -> None:
    mdp = zero_mean_variance_mdp(2.0)
    # u < .5 selects +2 and u >= .5 selects -2 for risky action.
    tapes = TapeBank(
        initial_u=np.full(100, 0.25),
        transition_u=np.concatenate(
            [np.full((50, 1), 0.25), np.full((50, 1), 0.75)],
            axis=0,
        ),
    )
    intended = exact_saa_posterior(mdp, tapes, beta=1.0)
    changed = exact_exponential_utility_posterior(mdp, tapes, beta=1.0)
    assert np.allclose(intended.probabilities, [0.5, 0.5])
    expected_risky_probability = math.cosh(2.0) / (1.0 + math.cosh(2.0))
    assert np.isclose(changed.probabilities[1], expected_risky_probability)
    assert changed.probabilities[1] > 0.75


def test_two_step_exact_returns_and_posterior() -> None:
    mdp = two_step_choice_mdp()
    posterior = exact_gibbs_posterior(mdp, beta=1.0)
    by_decisions = {
        tuple(policy[list(mdp.decision_states)]): value
        for policy, value in zip(posterior.policies, posterior.returns)
    }
    assert by_decisions[(0, 0)] == -0.25
    assert by_decisions[(0, 1)] == 2.0
    assert by_decisions[(1, 0)] == 0.5
    assert by_decisions[(1, 1)] == 0.5
    best_index = int(np.argmax(posterior.probabilities))
    assert tuple(posterior.policies[best_index, list(mdp.decision_states)]) == (0, 1)


def test_averaging_shared_single_tape_posteriors_is_not_expected_return_inference() -> None:
    # Safe return is 0; risky return is +3 (probability 1/4) or -1 (3/4).
    # The asymmetry matters: symmetric +/- rewards can hide this error.
    transition = np.zeros((3, 2, 3))
    reward = np.zeros_like(transition)
    transition[0, 0, 1] = 1.0
    transition[0, 1, 1:] = [0.25, 0.75]
    reward[0, 1, 1:] = [3.0, -1.0]
    transition[1:, :, 1] = 1.0
    mdp = FiniteHorizonMDP(
        transition, reward, np.array([1.0, 0.0, 0.0]), horizon=1,
        terminal=np.array([False, True, True]), decision_states=(0,),
    )
    expected = exact_gibbs_posterior(mdp)
    tapes = TapeBank(np.full(4, 0.1), np.array([[0.1], [0.5], [0.5], [0.5]]))
    conditional_probabilities = [
        exact_saa_posterior(mdp, TapeBank(np.array([0.1]), np.array([[u]]))).probabilities[1]
        for u in tapes.transition_u[:, 0]
    ]
    mixture = float(np.mean(conditional_probabilities))
    assert np.allclose(expected.probabilities, [0.5, 0.5])
    assert np.isclose(mixture, 0.25 / (1 + np.exp(-3)) + 0.75 / (1 + np.exp(1)))
    assert abs(mixture - 0.5) > 0.05
    # Averaging returns inside one fixed multi-tape target restores this example.
    assert np.allclose(exact_saa_posterior(mdp, tapes).probabilities, expected.probabilities)
