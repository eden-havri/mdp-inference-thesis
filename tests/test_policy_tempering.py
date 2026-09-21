from __future__ import annotations

import numpy as np
import pytest

from mdp_inference.exact import exact_gibbs_posterior
from mdp_inference.examples import two_step_choice_mdp
from mdp_inference.policy_tempering import (
    PolicyTemperingConfig,
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
    assert sampled.value_evaluations == 4 + 4 * 30_000
    assert np.all(sampled.local_acceptance_rates > 0.0)
    assert np.all(sampled.swap_acceptance_rates > 0.0)
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
