from __future__ import annotations

import numpy as np

from mdp_inference.exact import exact_gibbs_posterior
from mdp_inference.examples import two_step_choice_mdp
from mdp_inference.policy_mcmc import (
    PolicyMHConfig,
    autocorrelation_effective_sample_size,
    run_single_site_policy_mh,
)


def test_policy_mh_matches_exact_small_posterior_marginals() -> None:
    mdp = two_step_choice_mdp()
    beta = 1.4
    exact = exact_gibbs_posterior(mdp, beta=beta)
    initial = mdp.policy_from_decisions((0, 0))
    sampled = run_single_site_policy_mh(
        mdp,
        initial,
        PolicyMHConfig(
            beta=beta,
            iterations=60_000,
            burn_in=5_000,
            thinning=5,
            seed=41,
        ),
    )
    assert sampled.simulator_steps == 0
    assert sampled.value_evaluations + sampled.lazy_iterations == 60_001
    assert 0.0 < sampled.acceptance_rate < 1.0
    assert np.allclose(
        sampled.marginal_action_probabilities(mdp),
        exact.marginal_action_probabilities(mdp),
        atol=0.025,
    )


def test_policy_mh_samples_complete_committed_policies() -> None:
    mdp = two_step_choice_mdp()
    sampled = run_single_site_policy_mh(
        mdp,
        mdp.policy_from_decisions((1, 1)),
        PolicyMHConfig(iterations=100, burn_in=10, thinning=3, seed=2),
    )
    assert sampled.policies.shape == (30, mdp.num_states)
    assert np.all((sampled.policies >= 0) & (sampled.policies < mdp.num_actions))
    assert np.allclose(
        sampled.estimated_returns,
        [mdp.expected_return(policy) for policy in sampled.policies],
        atol=1e-12,
    )


def test_autocorrelation_ess_detects_repeated_samples() -> None:
    independent_like = np.asarray([-1.0, 1.0] * 50)
    sticky = np.repeat([-1.0, 1.0], 50)
    assert autocorrelation_effective_sample_size(independent_like) == len(independent_like)
    assert autocorrelation_effective_sample_size(sticky) < 5.0


def test_policy_mh_requires_and_accounts_for_laziness() -> None:
    mdp = two_step_choice_mdp()
    sampled = run_single_site_policy_mh(
        mdp,
        mdp.policy_from_decisions((0, 0)),
        PolicyMHConfig(
            iterations=500,
            burn_in=50,
            thinning=5,
            lazy_probability=0.2,
            seed=9,
        ),
    )
    assert 0 < sampled.lazy_iterations < 500
    assert sampled.value_evaluations + sampled.lazy_iterations == 501

    with np.testing.assert_raises(ValueError):
        run_single_site_policy_mh(
            mdp,
            mdp.policy_from_decisions((0, 0)),
            PolicyMHConfig(
                iterations=10,
                burn_in=1,
                lazy_probability=0.0,
            ),
        )
