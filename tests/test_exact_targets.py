from __future__ import annotations

import math

import numpy as np

from mdp_inference.exact import (
    exact_exponential_utility_posterior,
    exact_gibbs_posterior,
    exact_saa_posterior,
)
from mdp_inference.examples import two_step_choice_mdp, zero_mean_variance_mdp
from mdp_inference.mdp import TapeBank


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
