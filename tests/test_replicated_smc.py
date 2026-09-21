from __future__ import annotations

import numpy as np

from mdp_inference.exact import exact_saa_posterior
from mdp_inference.examples import two_step_choice_mdp, zero_mean_variance_mdp
from mdp_inference.mdp import FiniteHorizonMDP, TapeBank
from mdp_inference.replicated_smc import ReplicatedSMCConfig, run_replicated_policy_smc


def test_replicated_smc_matches_exact_saa_marginals() -> None:
    mdp = two_step_choice_mdp()
    tapes = TapeBank.sample(num_replicas=4, horizon=mdp.horizon, seed=4)
    exact = exact_saa_posterior(mdp, tapes, beta=1.0)
    result = run_replicated_policy_smc(
        mdp,
        tapes,
        ReplicatedSMCConfig(num_particles=20_000, beta=1.0, seed=5),
    )
    exact_marginals = exact.marginal_action_probabilities(mdp)
    sampled_marginals = result.marginal_action_probabilities(mdp)
    assert np.max(np.abs(exact_marginals - sampled_marginals)) < 0.035
    assert result.resampled[-1]
    assert np.isfinite(result.log_normalizer_estimate)


def test_more_replicas_approach_expected_return_target() -> None:
    mdp = zero_mean_variance_mdp(2.0)
    errors: dict[int, list[float]] = {1: [], 64: []}
    for seed in range(200):
        for replicas in errors:
            tapes = TapeBank.sample(replicas, mdp.horizon, seed=10_000 * replicas + seed)
            posterior = exact_saa_posterior(mdp, tapes)
            errors[replicas].append(abs(float(posterior.probabilities[1]) - 0.5))
    assert np.mean(errors[64]) < 0.3 * np.mean(errors[1])


def test_return_is_averaged_before_particle_weighting() -> None:
    mdp = zero_mean_variance_mdp(2.0)
    tapes = TapeBank(
        initial_u=np.array([0.1, 0.1]),
        transition_u=np.array([[0.25], [0.75]]),
    )
    result = run_replicated_policy_smc(
        mdp,
        tapes,
        ReplicatedSMCConfig(num_particles=15_000, beta=1.0, seed=10),
    )
    marginal = result.marginal_action_probabilities(mdp)[0]
    assert abs(float(marginal[1]) - 0.5) < 0.03


def test_smc_normalizer_is_unbiased_for_fixed_tapes() -> None:
    mdp = two_step_choice_mdp()
    tapes = TapeBank.sample(num_replicas=4, horizon=mdp.horizon, seed=4)
    exact_z = float(np.exp(exact_saa_posterior(mdp, tapes).log_normalizer))
    estimates = np.asarray(
        [
            np.exp(
                run_replicated_policy_smc(
                    mdp,
                    tapes,
                    ReplicatedSMCConfig(num_particles=32, beta=1.0, seed=seed),
                ).log_normalizer_estimate
            )
            for seed in range(600)
        ]
    )
    standard_error = float(estimates.std(ddof=1) / np.sqrt(len(estimates)))
    assert abs(float(estimates.mean()) - exact_z) < 4.0 * standard_error


def test_nonuniform_proposal_correction_matches_exact_fixed_tape_target() -> None:
    """A skewed proposal must not change the SAA posterior or normalizer.

    The sole decision state is revisited for all three steps.  Consequently the
    prior/proposal correction belongs in the particle weight exactly once, even
    though the selected action produces reward three times.  The proposal puts
    only 6% mass on the action receiving about 86% posterior mass, making a
    missing, repeated, or reversed correction fail by a wide margin.
    """

    transition = np.ones((1, 2, 1), dtype=np.float64)
    reward = np.zeros_like(transition)
    reward[0, 1, 0] = 0.4
    mdp = FiniteHorizonMDP(
        transition=transition,
        reward=reward,
        initial=np.array([1.0]),
        horizon=3,
        decision_states=(0,),
    )
    tapes = TapeBank(
        initial_u=np.array([0.1, 0.7, 0.4]),
        transition_u=np.array(
            [
                [0.1, 0.2, 0.3],
                [0.7, 0.8, 0.9],
                [0.4, 0.5, 0.6],
            ]
        ),
    )
    prior = np.array([[0.35, 0.65]], dtype=np.float64)
    proposal = np.array([[0.94, 0.06]], dtype=np.float64)

    exact = exact_saa_posterior(
        mdp,
        tapes,
        beta=1.0,
        prior_probabilities=prior,
    )
    expected_z = 0.35 + 0.65 * np.exp(1.2)
    expected_action_one = 0.65 * np.exp(1.2) / expected_z
    assert np.allclose(exact.returns, [0.0, 1.2])
    assert np.isclose(exact.log_normalizer, np.log(expected_z))
    assert np.isclose(exact.marginal_action_probabilities(mdp)[0, 1], expected_action_one)
    assert expected_action_one - proposal[0, 1] > 0.75

    result = run_replicated_policy_smc(
        mdp,
        tapes,
        ReplicatedSMCConfig(
            num_particles=5_000,
            beta=1.0,
            ess_threshold_ratio=0.5,
            seed=1_729,
        ),
        proposal_probabilities=proposal,
        prior_probabilities=prior,
    )

    sampled_marginals = result.marginal_action_probabilities(mdp)
    exact_marginals = exact.marginal_action_probabilities(mdp)
    # The first resampling copies the memoized action.  The correction must not
    # be applied again when that same state is revisited by its descendants.
    assert np.array_equal(result.resampled, [True, False, True])
    assert np.max(np.abs(sampled_marginals - exact_marginals)) < 0.01
    assert abs(result.log_normalizer_estimate - exact.log_normalizer) < 0.02
