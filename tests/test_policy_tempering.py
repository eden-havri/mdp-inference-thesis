from __future__ import annotations

import numpy as np
import pytest

from mdp_inference.exact import exact_gibbs_posterior
from mdp_inference.examples import two_step_choice_mdp
from mdp_inference.policy_tempering import (
    PolicyTemperingConfig,
    PolicyTemperingResult,
    power_beta_ladder,
    run_replica_exchange_policy_mh,
)


def test_power_beta_ladder_has_exact_endpoints() -> None:
    ladder = power_beta_ladder(beta=16.0, num_temperatures=5, power=2.0)
    assert ladder[0] == 0.0
    assert ladder[-1] == 16.0
    assert np.all(np.diff(ladder) > 0.0)


def test_replica_exchange_matches_exact_policy_marginals() -> None:
    mdp = two_step_choice_mdp()
    beta = 1.4
    exact = exact_gibbs_posterior(mdp, beta=beta)
    guide = np.full((mdp.num_states, mdp.num_actions), 0.5)
    guide[0] = [0.85, 0.15]
    sampled = run_replica_exchange_policy_mh(
        mdp,
        mdp.policy_from_decisions((0, 0)),
        PolicyTemperingConfig(
            beta=beta,
            num_temperatures=4,
            iterations=30_000,
            burn_in=3_000,
            thinning=3,
            swap_interval=1,
            guide_strength=0.7,
            seed=912,
        ),
        guide_probabilities=guide,
    )
    assert sampled.simulator_steps == 0
    assert sampled.lazy_iterations > 0
    assert np.all(sampled.local_proposals == 30_000 - sampled.lazy_iterations)
    assert sampled.value_evaluations == 4 + int(sampled.local_proposals.sum())
    assert np.all(sampled.local_acceptance_rates > 0.0)
    assert np.all(sampled.swap_acceptance_rates > 0.0)
    monitored_iterations = 30_000 - 3_000
    assert np.all(sampled.walker_temperature_visits.sum(axis=1) == monitored_iterations)
    assert sampled.cold_walker_count == 4
    assert sampled.walker_round_trips.sum() > 0
    assert np.allclose(
        sampled.marginal_action_probabilities(mdp),
        exact.marginal_action_probabilities(mdp),
        atol=0.025,
    )


def test_tempering_rejects_a_non_full_support_guide_strength() -> None:
    mdp = two_step_choice_mdp()
    with pytest.raises(ValueError, match="guide_strength"):
        run_replica_exchange_policy_mh(
            mdp,
            mdp.policy_from_decisions((0, 0)),
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                guide_strength=1.0,
            ),
            guide_probabilities=np.full(
                (mdp.num_decisions, mdp.num_actions),
                1.0 / mdp.num_actions,
            ),
        )


def test_tempering_requires_a_positive_whole_sweep_lazy_probability() -> None:
    mdp = two_step_choice_mdp()
    with pytest.raises(ValueError, match="lazy_probability"):
        run_replica_exchange_policy_mh(
            mdp,
            mdp.policy_from_decisions((0, 0)),
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                lazy_probability=0.0,
            ),
        )


def test_overdispersed_initialization_does_not_change_the_exact_target() -> None:
    mdp = two_step_choice_mdp()
    exact = exact_gibbs_posterior(mdp, beta=1.0)
    sampled = run_replica_exchange_policy_mh(
        mdp,
        mdp.policy_from_decisions((1, 1)),
        PolicyTemperingConfig(
            beta=1.0,
            num_temperatures=3,
            iterations=20_000,
            burn_in=2_000,
            thinning=3,
            prior_initialize_hot_replicas=True,
            seed=73,
        ),
    )
    assert np.allclose(
        sampled.marginal_action_probabilities(mdp),
        exact.marginal_action_probabilities(mdp),
        atol=0.035,
    )


def test_policy_occupancy_diagnostics_flag_a_stuck_decision_state() -> None:
    mdp = two_step_choice_mdp()
    policies = np.stack(
        [
            mdp.policy_from_decisions((0, 0)),
            mdp.policy_from_decisions((1, 0)),
            mdp.policy_from_decisions((0, 0)),
            mdp.policy_from_decisions((1, 0)),
        ]
    )
    sampled = PolicyTemperingResult(
        policies=policies,
        estimated_returns=np.ones(4),
        betas=np.asarray([0.0, 1.0]),
        local_proposals=np.ones(2, dtype=np.int64),
        local_accepts=np.ones(2, dtype=np.int64),
        swap_proposals=np.ones(1, dtype=np.int64),
        swap_accepts=np.ones(1, dtype=np.int64),
        walker_temperature_visits=np.asarray([[2, 2], [2, 2]]),
        walker_endpoint_transitions=np.asarray([1, 1]),
        walker_round_trips=np.asarray([0, 0]),
        lazy_iterations=0,
        value_evaluations=0,
        simulator_steps=0,
    )

    diagnostics = sampled.policy_occupancy_diagnostics(mdp)

    assert sampled.return_effective_sample_size == 4.0
    assert diagnostics.conservative_effective_sample_size == 0.0
    assert diagnostics.states_without_action_switches == 1
    assert diagnostics.unique_policy_count == 2
    assert diagnostics.unique_policy_fraction == 0.5
    assert diagnostics.state_action_switch_rates.tolist() == [1.0, 0.0]
    assert diagnostics.mean_policy_hamming_jump == 0.5
    assert diagnostics.state_action_effective_sample_sizes[0].tolist() == [4.0, 4.0]
    assert np.all(np.isnan(diagnostics.state_action_effective_sample_sizes[1]))


def test_walker_flow_accounting_is_post_burn_in_and_normalized() -> None:
    mdp = two_step_choice_mdp()
    iterations = 400
    burn_in = 100
    sampled = run_replica_exchange_policy_mh(
        mdp,
        mdp.policy_from_decisions((0, 0)),
        PolicyTemperingConfig(
            beta=0.01,
            num_temperatures=3,
            iterations=iterations,
            burn_in=burn_in,
            thinning=10,
            swap_interval=1,
            guide_strength=0.0,
            seed=112,
        ),
    )

    assert np.all(
        sampled.walker_temperature_visits.sum(axis=1) == iterations - burn_in
    )
    assert np.allclose(sampled.temperature_visit_fractions.sum(axis=1), 1.0)
    assert sampled.cold_walker_count == 3
    assert sampled.walker_round_trips.sum() > 0
    assert np.all(sampled.walker_endpoint_transitions >= 2 * sampled.walker_round_trips)
