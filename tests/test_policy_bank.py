from __future__ import annotations

import numpy as np

from mdp_inference.exact import _enumerated_policies, exact_gibbs_posterior, exact_saa_posterior
from mdp_inference.examples import two_step_choice_mdp
from mdp_inference.mdp import TapeBank
from mdp_inference.policy_bank import restricted_policy_bank_posterior


def test_complete_policy_bank_matches_exact_gibbs_posterior() -> None:
    mdp = two_step_choice_mdp()
    policies = _enumerated_policies(mdp)
    exact = exact_gibbs_posterior(mdp, beta=1.7)
    bank = restricted_policy_bank_posterior(mdp, policies, beta=1.7)
    assert np.array_equal(bank.policies, exact.policies)
    assert np.allclose(bank.estimated_returns, exact.returns, atol=1e-12)
    assert np.allclose(bank.probabilities, exact.probabilities, atol=1e-12)
    assert np.isclose(bank.restricted_log_normalizer, exact.log_normalizer, atol=1e-12)
    assert np.allclose(
        bank.marginal_action_probabilities(mdp),
        exact.marginal_action_probabilities(mdp),
        atol=1e-12,
    )


def test_policy_bank_deduplicates_support_without_multiplying_prior_mass() -> None:
    mdp = two_step_choice_mdp()
    policies = _enumerated_policies(mdp)
    duplicated = np.concatenate((policies, policies[[0, 0, 2]]), axis=0)
    unique_bank = restricted_policy_bank_posterior(mdp, policies, beta=0.9)
    duplicated_bank = restricted_policy_bank_posterior(mdp, duplicated, beta=0.9)
    assert np.array_equal(duplicated_bank.policies, unique_bank.policies)
    assert np.allclose(duplicated_bank.probabilities, unique_bank.probabilities, atol=1e-12)


def test_complete_policy_bank_matches_fixed_tape_saa_posterior() -> None:
    mdp = two_step_choice_mdp()
    policies = _enumerated_policies(mdp)
    tapes = TapeBank.sample(num_replicas=11, horizon=mdp.horizon, seed=19)
    exact = exact_saa_posterior(mdp, tapes, beta=1.3)
    bank = restricted_policy_bank_posterior(mdp, policies, beta=1.3, tapes=tapes)
    assert np.allclose(bank.estimated_returns, exact.returns, atol=1e-12)
    assert np.allclose(bank.probabilities, exact.probabilities, atol=1e-12)
    assert np.isclose(bank.restricted_log_normalizer, exact.log_normalizer, atol=1e-12)
    assert bank.simulator_steps > 0
